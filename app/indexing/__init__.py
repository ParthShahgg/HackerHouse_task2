"""Offline ingestion: dataset streaming -> normalise -> dedup -> chunk -> index.

Nothing in this package runs on the live query path. Chunking and embedding of
the corpus happen here, ahead of time, which is what makes the online path
short enough to be latency-engineered.
"""
