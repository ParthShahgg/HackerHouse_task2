"""Retrieval stack: BGE-M3 encoding, hybrid RRF search, reranking, gating."""

from app.retrieval.confidence import ConfidenceGate, GateThresholds, abstain_message
from app.retrieval.embedder import BGEM3Embedder, get_embedder
from app.retrieval.hybrid import HybridRetriever, decide_languages, reciprocal_rank_fusion
from app.retrieval.parent_expansion import expand_to_parents
from app.retrieval.reranker import BGEReranker, get_reranker
from app.retrieval.service import RetrievalService
from app.retrieval.store import QdrantStore, get_store

__all__ = [
    "BGEM3Embedder",
    "BGEReranker",
    "ConfidenceGate",
    "GateThresholds",
    "HybridRetriever",
    "QdrantStore",
    "RetrievalService",
    "abstain_message",
    "decide_languages",
    "expand_to_parents",
    "get_embedder",
    "get_reranker",
    "get_store",
    "reciprocal_rank_fusion",
]
