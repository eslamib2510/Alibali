"""Gemini calls. Two entry points: `generate_reply(...)` and `embed(...)`.

Kept deliberately thin so swapping to OpenAI/Claude later is a one-file change.

Everything here is SYNCHRONOUS and blocking. Async callers (the ARQ worker,
FastAPI routes) must dispatch via `asyncio.to_thread(...)` — calling directly
from a coroutine stalls the event loop for the duration of the HTTP round trip
and serializes every other job sharing that loop.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from google import genai
from google.genai import types

from app.config import settings

# One client instance reused across requests. The SDK is thread-safe.
_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def generate_reply(
    system_prompt: str,
    history: Iterable[tuple[str, str]],
    model: str = settings.RECEPTIONIST_DEFAULT_MODEL,
) -> tuple[str, dict]:
    """Generate the next assistant reply.

    Args:
        system_prompt: tenant's system prompt (persona + instructions).
        history: iterable of (role, text) tuples, oldest first.
                 role in {'user', 'assistant'}.
        model: Gemini model id.

    Returns:
        (reply_text, usage_metadata) where usage_metadata is whatever the SDK
        gave us (token counts). We log it but don't depend on the shape.
    """
    contents = []
    for role, text in history:
        # Gemini expects 'user' and 'model' as roles.
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(role=gemini_role, parts=[types.Part.from_text(text=text)])
        )

    response = _client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Conservative defaults; tune per tenant later.
            temperature=0.6,
            max_output_tokens=600,
        ),
    )

    text = (response.text or "").strip()
    usage = {
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
        "candidates_tokens": getattr(
            response.usage_metadata, "candidates_token_count", None
        ),
        "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
    }
    return text, usage


# ─── Embeddings (RAG) ────────────────────────────────────────────────────────

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMS = 1536  # must match knowledge.embedding vector(1536)

# Gemini's retrieval task types. Using the right one on each side is worth a
# few points of recall: stored chunks are embedded as documents, the customer's
# question as a query, and the model places them in the same space accordingly.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length.

    gemini-embedding-001 only pre-normalizes its native 3072-dim output. Any
    truncated size — including our 1536 — comes back UNNORMALIZED and Google
    documents that you must normalize it yourself.

    It matters more than it looks. pgvector's cosine operator normalizes
    internally so search still works either way, but raw magnitudes then leak
    into anything that doesn't: inner-product search (`<#>`), similarity
    thresholds, and score comparisons across chunks of different lengths.
    Normalizing once at write time keeps every downstream consumer honest.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed(
    text: str,
    *,
    task_type: str = TASK_DOCUMENT,
    dims: int = EMBEDDING_DIMS,
) -> list[float]:
    """Embed one string. Returns a normalized `dims`-length vector.

    Use `task_type=TASK_QUERY` when embedding a customer's question, and the
    default `TASK_DOCUMENT` when embedding knowledge to store.
    """
    return embed_batch([text], task_type=task_type, dims=dims)[0]


def embed_batch(
    texts: Sequence[str],
    *,
    task_type: str = TASK_DOCUMENT,
    dims: int = EMBEDDING_DIMS,
) -> list[list[float]]:
    """Embed many strings in one request — the cheap path for ingestion.

    Returns vectors in the same order as `texts`. Raises RuntimeError if the
    API returns a different number of embeddings than we asked for, rather
    than letting a silent misalignment attach the wrong vector to a chunk.
    """
    if not texts:
        return []

    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=list(texts),
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=dims,
        ),
    )

    embeddings = response.embeddings or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Gemini returned {len(embeddings)} embeddings for {len(texts)} "
            "inputs — refusing to guess the alignment"
        )

    return [_normalize(list(e.values or [])) for e in embeddings]
