"""Retrieval — the read side of RAG.

Finds the knowledge chunks relevant to a customer's question and hands them to
the agent as grounding context. Runs inside the live message path, so latency
matters here in a way it doesn't during ingestion.
"""
