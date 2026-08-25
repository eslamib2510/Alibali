"""Ingestion — the write side of RAG.

Turns raw source material into rows in `knowledge`: chunk it, embed it, store
it. Everything here runs on upload or on a sync job, never in a customer's
message path.

Kept apart from `app/retrieval/` because the two sides have genuinely
different shapes — ingestion is bursty, batch-oriented and allowed to be slow;
retrieval runs inside a live conversation and is not. If ingestion ever needs
its own worker process, the boundary is already drawn here.
"""
