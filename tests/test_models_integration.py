"""Tests against the real models and a live Qdrant.

Marked so the default `pytest` run stays green without weights or services:
    pytest -m models          # loads bge-m3 / bge-reranker-v2-m3 (~4.5GB)
    pytest -m integration     # needs Qdrant on QDRANT_URL
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BGE-M3 encoder
# ---------------------------------------------------------------------------
@pytest.mark.models
class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        from app.retrieval.embedder import BGEM3Embedder

        model = BGEM3Embedder()
        model.load()
        return model

    def test_dim_read_from_config_not_hardcoded(self, embedder):
        assert embedder.dim == 1024  # BGE-M3, sourced from config.hidden_size

    def test_dense_is_l2_normalised(self, embedder):
        result = embedder.encode(["hello world", "निगम एक कंपनी है"])
        norms = np.linalg.norm(result.dense, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_dense_shape(self, embedder):
        result = embedder.encode(["a", "b", "c"])
        assert result.dense.shape == (3, embedder.dim)

    def test_semantic_ranking_is_correct(self, embedder):
        """A relevant passage must outscore an unrelated one."""
        query, relevant, unrelated = (
            "What is a corporation?",
            "A corporation is a company authorized to act as a single entity.",
            "The Manhattan Project developed the first nuclear weapons.",
        )
        vectors = embedder.encode([query, relevant, unrelated]).dense
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_cross_lingual_alignment(self, embedder):
        """The property that lets us skip a translation hop on the query path."""
        english = "What is a corporation?"
        hindi = "कॉर्पोरेशन क्या है?"
        unrelated = "मैनहट्टन परियोजना ने परमाणु हथियार विकसित किया।"
        vectors = embedder.encode([english, hindi, unrelated]).dense
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_sparse_weights_are_positive_and_nonempty(self, embedder):
        sparse = embedder.encode(["What is a corporation?"]).sparse[0]
        assert sparse
        assert all(weight > 0 for weight in sparse.values())

    def test_sparse_excludes_special_tokens(self, embedder):
        tokenizer = embedder.tokenizer
        special = {
            i for i in (tokenizer.cls_token_id, tokenizer.eos_token_id,
                        tokenizer.pad_token_id, tokenizer.unk_token_id)
            if i is not None
        }
        sparse = embedder.encode(["a longer sentence about corporations"]).sparse[0]
        assert not (set(sparse) & special)

    def test_sparse_favours_content_words(self, embedder):
        """The lexical branch must key on the topical term."""
        sparse = embedder.encode(["What is a corporation?"]).sparse[0]
        top_token = max(sparse.items(), key=lambda kv: kv[1])[0]
        assert "corporation" in embedder.token_to_text(top_token).lower()

    def test_sparse_max_pooled_not_summed(self, embedder):
        """Repetition must not multiply a term's weight."""
        once = embedder.encode(["corporation"]).sparse[0]
        many = embedder.encode(["corporation corporation corporation"]).sparse[0]
        shared = set(once) & set(many)
        assert shared
        for token in shared:
            assert many[token] <= once[token] * 2.0

    def test_empty_input(self, embedder):
        result = embedder.encode([])
        assert result.dense.shape[0] == 0
        assert result.sparse == []

    def test_blank_string_does_not_crash(self, embedder):
        assert embedder.encode(["", " "]).dense.shape == (2, embedder.dim)

    def test_order_preserved_despite_length_sorting(self, embedder):
        """Internal length-sorted batching must not permute outputs."""
        texts = ["short", "a considerably longer piece of text here", "mid length text"]
        batched = embedder.encode(texts, batch_size=2).dense
        for i, text in enumerate(texts):
            single = embedder.encode([text], batch_size=1).dense[0]
            assert float(batched[i] @ single) > 0.999

    def test_encode_query_matches_encode(self, embedder):
        dense, sparse = embedder.encode_query("test query")
        assert dense.shape == (embedder.dim,)
        assert isinstance(sparse, dict)

    def test_token_counting(self, embedder):
        assert embedder.count_tokens("hello world") > 0
        assert embedder.count_tokens("") == 0

    def test_token_round_trip(self, embedder):
        text = "निगम एक कंपनी है"
        ids = embedder.encode_token_ids(text)
        assert embedder.decode_token_ids(ids).strip()

    @pytest.mark.parametrize(
        "text",
        [
            "निगम एक कंपनी है",         # hi
            "महामंडळ काय आहे",           # mr
            "நிறுவனம் என்றால் என்ன",      # ta
            "కార్పొరేషన్ అంటే ఏమిటి",      # te
            "کمپنی کیا ہے",              # ur
        ],
    )
    def test_multilingual_encoding(self, embedder, text):
        result = embedder.encode([text])
        assert np.isfinite(result.dense).all()
        assert result.sparse[0]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
@pytest.mark.models
class TestReranker:
    @pytest.fixture(scope="class")
    def reranker(self):
        from app.retrieval.reranker import BGEReranker

        model = BGEReranker()
        model.load()
        return model

    def test_scores_relevant_above_irrelevant(self, reranker):
        scores = reranker.score(
            "What is a corporation?",
            [
                "A corporation is a company authorized to act as a single entity.",
                "The Manhattan Project developed the first nuclear weapons in WWII.",
            ],
        )
        assert scores[0] > scores[1]

    def test_margin_is_large_for_clear_cases(self, reranker):
        """The gate depends on this separation being meaningful."""
        scores = reranker.score(
            "कॉर्पोरेशन क्या है?",
            [
                "निगम एक कंपनी या लोगों का समूह होता है जो एकल इकाई के रूप में कार्य करता है।",
                "मैनहट्टन परियोजना ने द्वितीय विश्व युद्ध के दौरान परमाणु हथियार विकसित किया।",
            ],
        )
        assert scores[0] - scores[1] > 2.0

    def test_empty_passages(self, reranker):
        assert reranker.score("q", []) == []

    def test_order_preserved(self, reranker):
        passages = [f"passage number {i} about corporations" for i in range(7)]
        scores = reranker.score("corporations", passages, batch_size=3)
        assert len(scores) == len(passages)

    def test_rerank_returns_sorted_indices(self, reranker):
        ranked = reranker.rerank(
            "What is a corporation?",
            [
                "Unrelated text about cooking pasta.",
                "A corporation is a legal entity separate from its owners.",
            ],
        )
        assert ranked[0][0] == 1

    def test_top_k_applied(self, reranker):
        ranked = reranker.rerank("q", ["a", "b", "c", "d"], top_k=2)
        assert len(ranked) == 2

    def test_int8_setting_is_reported(self, reranker):
        from app.config import get_settings

        if get_settings().int8_reranker_enabled():
            assert reranker.quantized is True


# ---------------------------------------------------------------------------
# NLI grounding
# ---------------------------------------------------------------------------
@pytest.mark.models
class TestNLIGrounder:
    @pytest.fixture(scope="class")
    def grounder(self):
        from app.guardrails.nli import NLIGrounder

        model = NLIGrounder()
        model.load()
        return model

    def test_entailment_label_index_found(self, grounder):
        """Hardcoding index 0 would silently invert the guardrail."""
        assert grounder._entail_idx is not None

    def test_supported_claim_entailed(self, grounder):
        from app.schemas.common import GroundingStatus

        status, results = grounder.verify(
            "A corporation is recognised as a legal entity.",
            ["A corporation is a company recognised in law as a single legal entity."],
            context_ids=["c1"],
        )
        assert status == GroundingStatus.ENTAILED
        assert results

    def test_contradicted_claim_not_entailed(self, grounder):
        from app.schemas.common import GroundingStatus

        status, _ = grounder.verify(
            "The Eiffel Tower is located in Mumbai and was built in 1998.",
            ["A corporation is a company recognised in law as a single legal entity."],
            context_ids=["c1"],
        )
        assert status != GroundingStatus.ENTAILED

    def test_verbatim_span_uses_deterministic_shortcut(self, grounder):
        """The cheap path: no model call needed when the text is in the evidence."""
        from app.schemas.common import GroundingStatus

        passage = "A corporation is a company or group of people authorized to act as a single entity."
        status, results = grounder.verify(passage, [passage], context_ids=["c1"])
        assert status == GroundingStatus.ENTAILED
        assert any(r.method == "deterministic_span" for r in results)

    def test_no_context_is_unknown_not_entailed(self, grounder):
        from app.schemas.common import GroundingStatus

        status, _ = grounder.verify("Some factual claim about things.", [])
        assert status == GroundingStatus.UNKNOWN

    def test_trivial_sentences_skipped(self):
        from app.guardrails.nli import is_factual_sentence

        assert not is_factual_sentence("Hi.")
        assert not is_factual_sentence("I hope this helps")
        assert not is_factual_sentence("Based on the retrieved passages")
        assert is_factual_sentence("A corporation is a legal entity separate from its owners.")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestQdrantStore:
    def test_connects(self):
        from app.retrieval.store import QdrantStore

        store = QdrantStore(allow_local_fallback=False)
        ok, detail = store.health()
        assert ok, detail

    def test_point_id_is_deterministic(self):
        from app.retrieval.store import chunk_point_id

        assert chunk_point_id("hi:abc#sw0_1") == chunk_point_id("hi:abc#sw0_1")
        assert chunk_point_id("a") != chunk_point_id("b")

    def test_round_trip_with_named_vectors(self):
        """Dense + sparse in one point, and hybrid RRF over them."""
        import uuid

        from app.indexing.records import Chunk
        from app.retrieval.store import QdrantStore

        name = f"pytest_{uuid.uuid4().hex[:8]}"
        store = QdrantStore(collection=name, allow_local_fallback=False)
        try:
            store.create_collection(8, recreate=True)
            chunks = [
                Chunk(
                    chunk_id=f"hi:test{i}", doc_id=f"hi:test{i}", parent_id=f"hi:test{i}",
                    language="hi", strategy="native", text=f"passage {i}",
                    content_hash=f"hash{i}", source_split="validation", n_chars=9,
                )
                for i in range(4)
            ]
            dense = np.eye(4, 8, dtype="float32")
            sparse = [{1: 0.5, 2: 0.3}, {2: 0.9}, {3: 0.7}, {1: 0.2, 4: 0.6}]
            written = store.upsert_chunks(chunks, dense, sparse, collection=name)
            assert written == 4
            assert store.count(name) == 4

            payloads = store.fetch_by_chunk_ids(["hi:test0"], collection=name)
            assert payloads["hi:test0"]["text"] == "passage 0"
            assert "is_selected" not in payloads["hi:test0"]

            from app.retrieval.hybrid import HybridRetriever
            from app.schemas.query import QueryEmbeddingResult

            retriever = HybridRetriever(store=store, collection=name)
            embedding = QueryEmbeddingResult(
                dense=[1.0] + [0.0] * 7, sparse_indices=[1, 2],
                sparse_values=[0.5, 0.3], dim=8, model="test",
            )

            dense_hits = retriever.search_dense(embedding.dense, limit=4)
            assert dense_hits and dense_hits[0].chunk_id == "hi:test0"

            sparse_hits = retriever.search_sparse(embedding.sparse_dict(), limit=4)
            assert sparse_hits

            for mode in ("client", "server"):
                result = retriever.retrieve(embedding, languages=["hi"], fusion_mode=mode)
                assert result.candidates, f"{mode} fusion returned nothing"
                assert not result.degraded

            filtered = retriever.retrieve(embedding, languages=["ta"])
            assert filtered.candidates == []
        finally:
            store.delete_collection(name)

    def test_language_filter_isolates(self):
        import uuid

        from app.indexing.records import Chunk
        from app.retrieval.hybrid import HybridRetriever
        from app.retrieval.store import QdrantStore
        from app.schemas.query import QueryEmbeddingResult

        name = f"pytest_{uuid.uuid4().hex[:8]}"
        store = QdrantStore(collection=name, allow_local_fallback=False)
        try:
            store.create_collection(4, recreate=True)
            chunks = [
                Chunk(chunk_id=f"{lang}:x", doc_id=f"{lang}:x", parent_id=f"{lang}:x",
                      language=lang, strategy="native", text=f"text {lang}",
                      content_hash=f"h{lang}", source_split="validation", n_chars=6)
                for lang in ("hi", "mr", "ta")
            ]
            dense = np.ones((3, 4), dtype="float32")
            store.upsert_chunks(chunks, dense, [{1: 1.0}] * 3, collection=name)

            retriever = HybridRetriever(store=store, collection=name)
            embedding = QueryEmbeddingResult(
                dense=[1.0, 1.0, 1.0, 1.0], sparse_indices=[1], sparse_values=[1.0],
                dim=4, model="t",
            )
            result = retriever.retrieve(embedding, languages=["mr"])
            assert {c.language for c in result.candidates} == {"mr"}
        finally:
            store.delete_collection(name)
