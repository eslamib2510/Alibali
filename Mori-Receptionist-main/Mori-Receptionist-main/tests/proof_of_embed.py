"""Embedding harness.

    python -m tests.proof_of_embed

The Gemini call itself is stubbed — no API key needed and no network. What's
under test is our own handling around it, which is where the bugs would be:

  1. Vectors come back normalized. gemini-embedding-001 only pre-normalizes
     its native 3072 dims; our truncated 1536 arrives unnormalized and Google
     documents that callers must normalize. Getting this wrong is invisible —
     pgvector's cosine operator normalizes internally, so search still
     "works" while thresholds and inner-product queries quietly misbehave.
  2. The right dimensionality and task type reach the API. A mismatch against
     the vector(1536) column is a runtime insert failure.
  3. A short/long response raises instead of silently misaligning vectors with
     chunks — the worst possible failure, since every answer would then be
     grounded in the wrong document.
  4. Empty input short-circuits without calling the API.
"""

from __future__ import annotations

import math
from unittest.mock import patch

from app.integrations import gemini

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeResponse:
    def __init__(self, vectors):
        self.embeddings = [FakeEmbedding(v) for v in vectors]


def stub(vectors):
    """Patch the SDK call, capturing the kwargs it was invoked with."""
    captured = {}

    def fake_embed_content(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return FakeResponse(vectors)

    return patch.object(
        gemini._client.models, "embed_content", fake_embed_content
    ), captured


def main() -> None:
    print("=" * 72)
    print("EMBEDDING HARNESS")
    print("=" * 72)

    # 1. Normalization — feed a deliberately unnormalized vector.
    raw = [3.0, 4.0] + [0.0] * 1534          # magnitude 5, not 1
    ctx, captured = stub([raw])
    with ctx:
        vec = gemini.embed("ice baths")
    length = math.sqrt(sum(v * v for v in vec))
    record(
        "Vector is normalized to unit length",
        abs(length - 1.0) < 1e-9 and abs(vec[0] - 0.6) < 1e-9,
        f"input magnitude 5.0 -> output magnitude {length:.12f} (want 1.0), "
        f"first component {vec[0]:.4f} (want 0.6)",
    )

    # 2. Correct model, dims and task type reach the API.
    ok = (
        captured["model"] == "gemini-embedding-001"
        and captured["config"].output_dimensionality == 1536
        and captured["config"].task_type == "RETRIEVAL_DOCUMENT"
    )
    record(
        "Model, dimensionality and task type are sent correctly",
        ok,
        f"model={captured['model']!r} "
        f"dims={captured['config'].output_dimensionality} (want 1536, matching "
        f"the vector(1536) column) "
        f"task_type={captured['config'].task_type!r}",
    )

    # 3. Query embeddings use the query task type, not the document one.
    ctx, captured = stub([[1.0] + [0.0] * 1535])
    with ctx:
        gemini.embed("do you sell cold plunges?", task_type=gemini.TASK_QUERY)
    record(
        "Query side uses RETRIEVAL_QUERY",
        captured["config"].task_type == "RETRIEVAL_QUERY",
        f"task_type={captured['config'].task_type!r} (want 'RETRIEVAL_QUERY')",
    )

    # 4. Batch preserves order and normalizes every vector.
    ctx, captured = stub([[3.0, 4.0] + [0.0] * 1534, [0.0, 5.0] + [0.0] * 1534])
    with ctx:
        vecs = gemini.embed_batch(["first", "second"])
    lengths = [math.sqrt(sum(v * v for v in vec)) for vec in vecs]
    ok = (
        len(vecs) == 2
        and all(abs(length - 1.0) < 1e-9 for length in lengths)
        and abs(vecs[0][0] - 0.6) < 1e-9      # first vector kept its direction
        and abs(vecs[1][1] - 1.0) < 1e-9      # second vector kept its direction
        and captured["contents"] == ["first", "second"]
    )
    record(
        "Batch returns one normalized vector per input, in order",
        ok,
        f"{len(vecs)} vectors, magnitudes {[round(x, 6) for x in lengths]}, "
        f"directions preserved",
    )

    # 5. Count mismatch must raise, never silently misalign.
    ctx, _ = stub([[1.0] + [0.0] * 1535])     # one vector for two inputs
    raised = None
    with ctx:
        try:
            gemini.embed_batch(["first", "second"])
        except RuntimeError as e:
            raised = str(e)
    record(
        "Mismatched embedding count raises instead of misaligning",
        raised is not None,
        f"raised: {raised!r}" if raised
        else "NO ERROR — vectors would attach to the wrong chunks",
    )

    # 6. Empty input never hits the API.
    called = {"yes": False}

    def should_not_run(**_kwargs):
        called["yes"] = True
        raise AssertionError("API called for empty input")

    with patch.object(gemini._client.models, "embed_content", should_not_run):
        empty = gemini.embed_batch([])
    record(
        "Empty input short-circuits without an API call",
        empty == [] and not called["yes"],
        f"returned {empty!r}, api_called={called['yes']}",
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
