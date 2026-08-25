"""Sentence-aware text chunking for the RAG ingest path.

Splits a document into overlapping chunks small enough to embed and to paste
into a prompt, without cutting sentences in half. A chunk that ends mid-clause
embeds badly and reads badly when the model quotes it back to a customer.

Design notes
------------
Token counting is approximated, not exact. Calling Gemini's count_tokens per
candidate chunk would mean an API round trip per sentence during ingestion;
`tiktoken` is an OpenAI tokenizer and would be wrong for Gemini anyway. We use
the standard ~4-characters-per-token heuristic, which is close enough for
English when the only consumer is a size budget with slack on both sides.
Chunks land near the target rather than exactly on it, which is fine — nothing
downstream requires precision.

Overlap is measured in whole sentences, not tokens. Carrying a partial
sentence into the next chunk would reintroduce the mid-sentence cut this
module exists to avoid, so we carry back whole sentences until the overlap
budget is met.

Written from scratch rather than vendored, so there's no third-party code or
license to track — `rag_plan.md` suggested copying LlamaIndex's SentenceSplitter,
but the useful part is ~40 lines of greedy packing.
"""

from __future__ import annotations

import re

# Average characters per token for English text. Gemini doesn't publish a
# cheap local tokenizer; this heuristic is the industry-standard stand-in.
CHARS_PER_TOKEN = 4

DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

# Sentence boundary: .!? followed by whitespace and a capital/quote/digit.
# The lookbehinds exclude the most common English abbreviations so "Dr. Smith"
# and "e.g. this" don't split. Note each pattern includes its trailing period:
# `(?<=[.!?])` is zero-width, so by the time these are evaluated the match
# position sits AFTER the dot, and a lookbehind of `\bDr` would be testing the
# wrong two characters. Deliberately not a full NLP sentence tokenizer — an
# occasional bad split costs a little retrieval quality, not correctness, and
# isn't worth an NLP dependency.
_ABBREVIATIONS = (
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)"
    r"(?<!\bJr\.)(?<!\bSr\.)(?<!\bvs\.)(?<!\bNo\.)(?<!\bInc\.)"
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)(?<!\bapprox\.)"
)
_SENTENCE_END = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")


def estimate_tokens(text: str) -> int:
    """Approximate token count. See module docstring on why this isn't exact."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving paragraph breaks as boundaries.

    Paragraphs are split first: a blank line is a hard boundary regardless of
    punctuation, which keeps list items and headings — common in FAQ and
    product copy, and often lacking terminal punctuation — from being glued
    onto the following sentence.
    """
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _split_oversized(sentence: str, max_tokens: int) -> list[str]:
    """Hard-split a single sentence that exceeds the budget on its own.

    Rare — it takes roughly 2000 characters without terminal punctuation — but
    it does happen with pasted tables, long URLs, or unpunctuated copy. We
    break on word boundaries so we never cut a word in half.
    """
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if current and estimate_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split `text` into overlapping, sentence-aligned chunks.

    Short input comes back as a single chunk — we never pad to reach the
    target. A three-line FAQ answer should stay one clean chunk rather than
    being merged with an unrelated neighbour.

    Args:
        text: the document to split.
        chunk_tokens: target size. Chunks land near it, not exactly on it.
        overlap_tokens: how much of the tail of each chunk to repeat at the
            head of the next, so an answer spanning a boundary is still
            retrievable from either side.

    Returns:
        Chunks in document order. Empty list for empty input.
    """
    if not text or not text.strip():
        return []
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if estimate_tokens(sentence) > chunk_tokens:
            sentences.extend(_split_oversized(sentence, chunk_tokens))
        else:
            sentences.append(sentence)

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current and current_tokens + sentence_tokens > chunk_tokens:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(s) for s in current)

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing sentences worth roughly `overlap_tokens`.

    Never returns the whole chunk. A one-sentence chunk therefore gets no
    overlap at all: carrying its only sentence forward would re-emit it as the
    head of the next chunk, doubling that chunk's size and, in the worst case,
    never advancing. Losing overlap on a chunk that is a single 500-token
    sentence costs nothing anyway — there's no boundary inside it for an
    answer to straddle.
    """
    if overlap_tokens <= 0 or len(sentences) <= 1:
        return []

    tail: list[str] = []
    total = 0
    # Walk backwards over everything except the first sentence, so at least
    # one sentence is always left behind and the next chunk advances.
    for sentence in reversed(sentences[1:]):
        tail.insert(0, sentence)
        total += estimate_tokens(sentence)
        if total >= overlap_tokens:
            break

    return tail
