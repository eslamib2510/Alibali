"""Chunker harness.

    python -m tests.proof_of_chunker

No database, no network. Checks the properties that actually matter for
retrieval quality, plus the two ways a greedy chunker can fail badly:
producing chunks that exceed the budget, or failing to advance and looping
forever.
"""

from __future__ import annotations

from app.ingestion.chunker import (
    DEFAULT_CHUNK_TOKENS,
    chunk_text,
    estimate_tokens,
    split_sentences,
)

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def main() -> None:
    print("=" * 72)
    print("CHUNKER HARNESS")
    print("=" * 72)

    # 1. Short input stays one chunk — never padded or merged.
    short = "We open at 9am. We close at 6pm."
    chunks = chunk_text(short)
    record(
        "Short text stays a single chunk",
        chunks == [short],
        f"{len(chunks)} chunk(s): {chunks!r}",
    )

    # 2. Empty and whitespace-only input produce nothing.
    blank = "   \n\n  "
    record(
        "Empty input returns no chunks",
        chunk_text("") == [] and chunk_text(blank) == [],
        f"empty -> {chunk_text('')!r}, whitespace -> {chunk_text(blank)!r}",
    )

    # 3. Sentences are never cut in half. Build a long doc of numbered
    #    sentences, then confirm every chunk starts and ends on a boundary.
    sentences = [f"This is sentence number {i} about ice baths and recovery." for i in range(1, 121)]
    doc = " ".join(sentences)
    chunks = chunk_text(doc)
    clean_edges = all(
        c.endswith(".") and c[0].isupper() for c in chunks
    )
    record(
        "No chunk starts or ends mid-sentence",
        clean_edges and len(chunks) > 1,
        f"{len(chunks)} chunks, all ending on '.' and starting capitalised: "
        f"{clean_edges}",
    )

    # 4. Chunks respect the budget. A greedy packer that appends before
    #    checking would blow past it.
    sizes = [estimate_tokens(c) for c in chunks]
    within = all(s <= DEFAULT_CHUNK_TOKENS * 1.15 for s in sizes)
    record(
        "Chunks stay within the token budget",
        within,
        f"target {DEFAULT_CHUNK_TOKENS}, actual sizes min={min(sizes)} "
        f"max={max(sizes)} (allowing 15% slack for the last sentence)",
    )

    # 5. Consecutive chunks overlap — an answer on a boundary is findable
    #    from either side.
    overlaps = []
    for a, b in zip(chunks, chunks[1:]):
        tail_words = set(a.split()[-40:])
        head_words = set(b.split()[:40])
        overlaps.append(len(tail_words & head_words) > 0)
    record(
        "Consecutive chunks share overlapping text",
        all(overlaps),
        f"{sum(overlaps)}/{len(overlaps)} chunk boundaries overlap",
    )

    # 6. Full coverage — no sentence is dropped.
    missing = [s for s in sentences if not any(s in c for c in chunks)]
    record(
        "Every sentence survives into at least one chunk",
        not missing,
        f"{len(sentences) - len(missing)}/{len(sentences)} sentences present"
        + (f", MISSING: {missing[:2]}" if missing else ""),
    )

    # 7. A single sentence bigger than the whole budget gets word-split
    #    rather than emitted oversized or dropped.
    giant = "word " * 3000  # ~3000 tokens, no sentence-ending punctuation
    chunks = chunk_text(giant)
    sizes = [estimate_tokens(c) for c in chunks]
    # Check each chunk separately: joining them would glue one chunk's last
    # word to the next chunk's first word and fake a split word.
    no_split_words = all("wordword" not in c for c in chunks)
    record(
        "Oversized single sentence is split on word boundaries",
        len(chunks) > 1
        and all(s <= DEFAULT_CHUNK_TOKENS * 1.15 for s in sizes)
        and no_split_words,
        f"{len(chunks)} chunks, max size {max(sizes)}, no words cut in half: "
        f"{no_split_words}",
    )

    # 8. Abbreviations don't cause false splits.
    abbrev = "Dr. Smith runs the clinic. Visit us at 9am, e.g. before work."
    sents = split_sentences(abbrev)
    record(
        "Common abbreviations don't trigger a split",
        len(sents) == 2,
        f"{len(sents)} sentences: {sents!r} (want 2 — 'Dr.' and 'e.g.' must "
        f"not split)",
    )

    # 9. Paragraph breaks are boundaries even without punctuation — headings
    #    and list items are common in FAQ copy.
    para = "Opening Hours\n\nWe open at 9am\n\nPricing\n\nFrom $50"
    sents = split_sentences(para)
    record(
        "Blank lines split even without terminal punctuation",
        len(sents) == 4,
        f"{len(sents)} pieces: {sents!r} (want 4)",
    )

    # 10. Termination. A pathological doc of many same-size sentences must not
    #     loop forever — if _overlap_tail ever returned the whole chunk, the
    #     packer would never advance.
    pathological = ". ".join([f"Sentence {i} here" for i in range(400)]) + "."
    chunks = chunk_text(pathological, chunk_tokens=60, overlap_tokens=50)
    record(
        "Aggressive overlap still terminates and advances",
        len(chunks) > 1 and chunks[0] != chunks[1],
        f"{len(chunks)} chunks with overlap 50 of budget 60; first two differ: "
        f"{chunks[0] != chunks[1]}",
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
