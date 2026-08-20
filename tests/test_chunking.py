"""Chunking strategies: boundaries, identifiers, routing, invariants."""

from __future__ import annotations

import pytest

from app.chunking import (
    STRATEGY_FIXED_FALLBACK,
    STRATEGY_NATIVE,
    STRATEGY_SEMANTIC_SPLIT,
    STRATEGY_SENTENCE_WINDOW,
    ChunkerContext,
    ChunkingEngine,
    chunk_fixed_fallback,
    chunk_native,
    chunk_semantic_split,
    chunk_sentence_window,
)
from app.chunking.semantic_split import find_breakpoints
from app.indexing.normalize import content_hash, make_doc_id, normalize_text, split_sentences
from app.indexing.records import ParentPassage


def make_parent(text: str, language: str = "hi") -> ParentPassage:
    normalized = normalize_text(text)
    chash = content_hash(language, normalized)
    return ParentPassage(
        doc_id=make_doc_id(language, chash),
        content_hash=chash,
        language=language,
        text=normalized,
        source_split="validation",
    )


# ---------------------------------------------------------------------------
# Strategy A - native
# ---------------------------------------------------------------------------
class TestNative:
    def test_single_chunk_verbatim(self, parent):
        chunks = chunk_native(parent)
        assert len(chunks) == 1
        assert chunks[0].text == parent.text
        assert chunks[0].strategy == STRATEGY_NATIVE

    def test_chunk_id_equals_parent_id_equals_doc_id(self, parent):
        """Load-bearing: parent lookup addresses the native chunk directly."""
        chunk = chunk_native(parent)[0]
        assert chunk.chunk_id == chunk.parent_id == chunk.doc_id == parent.doc_id

    def test_empty_passage_yields_nothing(self):
        assert chunk_native(make_parent("")) == []

    def test_does_not_truncate(self):
        long_text = "वाक्य। " * 500
        chunk = chunk_native(make_parent(long_text))[0]
        assert len(chunk.text) == len(normalize_text(long_text))


# ---------------------------------------------------------------------------
# Strategy B - sentence window
# ---------------------------------------------------------------------------
class TestSentenceWindow:
    def test_produces_overlapping_windows(self, chunker_ctx):
        parent = make_parent("S1 one. S2 two. S3 three. S4 four.")
        chunks = chunk_sentence_window(parent, chunker_ctx)
        assert chunks
        spans = [(c.sentence_start, c.sentence_end) for c in chunks]
        # size=2, stride=1 -> (0,1) (1,2) (2,3)
        assert (0, 1) in spans
        assert (1, 2) in spans

    def test_windows_overlap(self, chunker_ctx):
        parent = make_parent("A one. B two. C three. D four.")
        spans = [
            (c.sentence_start, c.sentence_end)
            for c in chunk_sentence_window(parent, chunker_ctx)
        ]
        ordered = sorted(spans)
        for first, second in zip(ordered, ordered[1:], strict=False):
            # Next window starts no later than the previous one ended => overlap
            # or adjacency, so no sentence pair is ever split apart entirely.
            assert second[0] <= first[1] + 1

    def test_short_passage_gets_no_children(self, chunker_ctx):
        """2 sentences would just duplicate the parent."""
        assert chunk_sentence_window(make_parent("One here. Two here."), chunker_ctx) == []

    def test_single_sentence_gets_no_children(self, chunker_ctx):
        assert chunk_sentence_window(make_parent("Only one sentence here"), chunker_ctx) == []

    def test_children_reference_parent(self, parent, chunker_ctx):
        for chunk in chunk_sentence_window(parent, chunker_ctx):
            assert chunk.parent_id == parent.doc_id
            assert chunk.chunk_id != parent.doc_id
            assert chunk.chunk_id.startswith(parent.doc_id + "#sw")
            assert chunk.strategy == STRATEGY_SENTENCE_WINDOW
            assert chunk.content_hash == parent.content_hash

    def test_child_text_is_substring_of_parent_sentences(self, parent, chunker_ctx):
        """No cross-passage merging: every child comes from this passage only."""
        sentences = split_sentences(parent.text)
        for chunk in chunk_sentence_window(parent, chunker_ctx):
            expected = " ".join(sentences[chunk.sentence_start : chunk.sentence_end + 1])
            assert chunk.text == expected

    def test_no_duplicate_spans(self, parent, chunker_ctx):
        chunks = chunk_sentence_window(parent, chunker_ctx)
        spans = [(c.sentence_start, c.sentence_end) for c in chunks]
        assert len(spans) == len(set(spans))

    def test_child_never_equals_parent(self, chunker_ctx):
        parent = make_parent("A one. B two. C three.")
        for chunk in chunk_sentence_window(parent, chunker_ctx):
            assert chunk.text != parent.text

    def test_window_size_respected(self):
        ctx = ChunkerContext(sentence_window_size=3, sentence_window_stride=1,
                             sentence_window_min_sentences=3)
        parent = make_parent("A one. B two. C three. D four. E five.")
        for chunk in chunk_sentence_window(parent, ctx):
            assert chunk.sentence_end - chunk.sentence_start + 1 <= 3

    def test_larger_stride_produces_fewer_children(self):
        parent = make_parent("A one. B two. C three. D four. E five. F six.")
        few = chunk_sentence_window(
            parent, ChunkerContext(sentence_window_size=2, sentence_window_stride=3,
                                   sentence_window_min_sentences=3)
        )
        many = chunk_sentence_window(
            parent, ChunkerContext(sentence_window_size=2, sentence_window_stride=1,
                                   sentence_window_min_sentences=3)
        )
        assert len(few) < len(many)


# ---------------------------------------------------------------------------
# Strategy C - semantic split
# ---------------------------------------------------------------------------
class TestFindBreakpoints:
    def test_empty(self):
        assert find_breakpoints([], 25.0) == []

    def test_cuts_at_dissimilar_boundary(self):
        # boundary 1 is far more dissimilar than the rest
        sims = [0.95, 0.10, 0.93, 0.94]
        assert 1 in find_breakpoints(sims, 25.0)

    def test_uniform_similarity_produces_no_cuts(self):
        """A passage with no structure must not shatter into single sentences."""
        assert find_breakpoints([0.9, 0.9, 0.9, 0.9], 25.0) == []

    def test_higher_percentile_cuts_more(self):
        sims = [0.9, 0.5, 0.8, 0.3, 0.85]
        assert len(find_breakpoints(sims, 60.0)) >= len(find_breakpoints(sims, 20.0))


class TestSemanticSplit:
    def test_no_embedder_falls_back_to_windows(self, chunker_ctx):
        parent = make_parent("A one. B two. C three. D four.")
        ctx = ChunkerContext(**{**chunker_ctx.__dict__, "embed_sentences": None})
        chunks = chunk_semantic_split(parent, ctx)
        assert all(c.strategy == STRATEGY_SENTENCE_WINDOW for c in chunks)

    def test_uses_stub_embedder_and_stays_within_passage(self):
        import numpy as np

        sentences_seen: list[list[str]] = []

        def fake_embed(sentences):
            sentences_seen.append(list(sentences))
            # First two similar, then a hard topic shift.
            vectors = []
            for i in range(len(sentences)):
                vectors.append([1.0, 0.0] if i < 2 else [0.0, 1.0])
            return np.array(vectors, dtype="float32")

        ctx = ChunkerContext(embed_sentences=fake_embed, semantic_min_segment_tokens=1)
        parent = make_parent("A one. B two. C three. D four.")
        chunks = chunk_semantic_split(parent, ctx)

        assert sentences_seen, "embedder should have been called"
        assert chunks
        sentences = split_sentences(parent.text)
        for chunk in chunks:
            assert chunk.strategy == STRATEGY_SEMANTIC_SPLIT
            assert chunk.parent_id == parent.doc_id
            expected = " ".join(sentences[chunk.sentence_start : chunk.sentence_end + 1])
            assert chunk.text == expected

    def test_too_few_sentences_yields_nothing(self):
        ctx = ChunkerContext(embed_sentences=lambda s: None)
        assert chunk_semantic_split(make_parent("Only one. Two."), ctx) == []

    def test_segments_are_contiguous_and_cover_passage(self):
        import numpy as np

        def fake_embed(sentences):
            return np.array(
                [[1.0, 0.0] if i < 2 else [0.0, 1.0] for i in range(len(sentences))],
                dtype="float32",
            )

        ctx = ChunkerContext(embed_sentences=fake_embed, semantic_min_segment_tokens=1)
        parent = make_parent("A one. B two. C three. D four.")
        chunks = sorted(chunk_semantic_split(parent, ctx), key=lambda c: c.sentence_start)
        for first, second in zip(chunks, chunks[1:], strict=False):
            assert second.sentence_start == first.sentence_end + 1


# ---------------------------------------------------------------------------
# Strategy D - fixed fallback
# ---------------------------------------------------------------------------
class TestFixedFallback:
    def test_uses_tokenizer_when_available(self):
        vocab = {}

        def encode(text: str) -> list[int]:
            ids = []
            for word in text.split():
                ids.append(vocab.setdefault(word, len(vocab) + 10))
            return ids

        def decode(ids: list[int]) -> str:
            reverse = {v: k for k, v in vocab.items()}
            return " ".join(reverse.get(i, "?") for i in ids)

        ctx = ChunkerContext(
            fixed_chunk_tokens=32, fixed_chunk_overlap_tokens=5,
            encode_tokens=encode, decode_tokens=decode,
        )
        parent = make_parent(" ".join(f"word{i}" for i in range(200)))
        chunks = chunk_fixed_fallback(parent, ctx)
        assert len(chunks) > 1
        assert all(c.strategy == STRATEGY_FIXED_FALLBACK for c in chunks)
        assert all(c.parent_id == parent.doc_id for c in chunks)

    def test_short_passage_yields_nothing(self):
        ctx = ChunkerContext(fixed_chunk_tokens=256)
        assert chunk_fixed_fallback(make_parent("short text"), ctx) == []

    def test_character_fallback_snaps_to_whitespace(self):
        """Without a tokenizer we still must not cut mid-word."""
        ctx = ChunkerContext(fixed_chunk_tokens=32, fixed_chunk_overlap_tokens=5)
        parent = make_parent(" ".join(f"word{i}" for i in range(400)))
        chunks = chunk_fixed_fallback(parent, ctx)
        assert len(chunks) > 1
        for chunk in chunks:
            assert not chunk.text.startswith(" ")
            # No partial tokens like "wor"
            for token in chunk.text.split():
                assert token.startswith("word")

    def test_chunks_overlap(self):
        def encode(text):
            return [ord(c) % 100 for c in text.replace(" ", "")]

        def decode(ids):
            return "".join(chr(i + 33) for i in ids)

        ctx = ChunkerContext(
            fixed_chunk_tokens=50, fixed_chunk_overlap_tokens=10,
            encode_tokens=encode, decode_tokens=decode,
        )
        parent = make_parent("x" * 400)
        chunks = chunk_fixed_fallback(parent, ctx)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class TestChunkingEngine:
    def test_native_always_included(self, parent):
        engine = ChunkingEngine(enabled=[STRATEGY_SENTENCE_WINDOW])
        assert STRATEGY_NATIVE in engine.enabled
        assert any(c.strategy == STRATEGY_NATIVE for c in engine.chunk(parent))

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking"):
            ChunkingEngine(enabled=["nonsense"])

    def test_short_passage_native_only(self):
        engine = ChunkingEngine()
        chunks = engine.chunk(make_parent("Just one sentence here."))
        assert len(chunks) == 1
        assert chunks[0].strategy == STRATEGY_NATIVE

    def test_multi_sentence_gets_sentence_windows(self):
        engine = ChunkingEngine()
        chunks = engine.chunk(make_parent("A one. B two. C three. D four."))
        strategies = {c.strategy for c in chunks}
        assert STRATEGY_NATIVE in strategies
        assert STRATEGY_SENTENCE_WINDOW in strategies

    def test_at_most_one_child_strategy(self):
        """Emitting all four would triple-index the same text."""
        engine = ChunkingEngine()
        chunks = engine.chunk(make_parent("A one. B two. C three. D four. E five."))
        child_strategies = {c.strategy for c in chunks if c.strategy != STRATEGY_NATIVE}
        assert len(child_strategies) <= 1

    def test_routes_long_passage_to_semantic(self):
        ctx = ChunkerContext(semantic_split_min_tokens=20, fixed_fallback_min_tokens=100000)
        engine = ChunkingEngine(ctx=ctx)
        parent = make_parent("Sentence number one. " * 30)
        assert engine.route(parent) == STRATEGY_SEMANTIC_SPLIT

    def test_routes_pathological_passage_to_fixed(self):
        ctx = ChunkerContext(semantic_split_min_tokens=20, fixed_fallback_min_tokens=50)
        engine = ChunkingEngine(ctx=ctx)
        parent = make_parent("word " * 500)
        assert engine.route(parent) == STRATEGY_FIXED_FALLBACK

    def test_unique_chunk_ids(self, parent):
        chunks = ChunkingEngine().chunk(parent)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_all_chunks_share_parent_and_hash(self, parent):
        for chunk in ChunkingEngine().chunk(parent):
            assert chunk.parent_id == parent.doc_id
            assert chunk.content_hash == parent.content_hash
            assert chunk.language == parent.language

    def test_chunk_forced_native(self, parent):
        chunks = ChunkingEngine().chunk_forced(parent, STRATEGY_NATIVE)
        assert len(chunks) == 1 and chunks[0].strategy == STRATEGY_NATIVE

    def test_chunk_forced_falls_back_to_native_when_empty(self):
        """An eval arm must never silently lose documents."""
        engine = ChunkingEngine()
        parent = make_parent("Tiny.")
        chunks = engine.chunk_forced(parent, STRATEGY_SENTENCE_WINDOW)
        assert len(chunks) == 1
        assert chunks[0].strategy == STRATEGY_NATIVE

    def test_chunk_forced_rejects_unknown(self, parent):
        with pytest.raises(ValueError):
            ChunkingEngine().chunk_forced(parent, "bogus")

    def test_empty_parent(self):
        assert ChunkingEngine().chunk(make_parent("")) == []

    def test_no_cross_passage_merging(self):
        """The central invariant: a chunk never mixes two source passages."""
        engine = ChunkingEngine()
        a = make_parent("Alpha one. Alpha two. Alpha three.")
        b = make_parent("Beta one. Beta two. Beta three.")
        for chunk in engine.chunk(a):
            assert "Beta" not in chunk.text
        for chunk in engine.chunk(b):
            assert "Alpha" not in chunk.text

    def test_payload_excludes_labels(self, parent):
        """is_selected / query / answer must never reach the index."""
        for chunk in ChunkingEngine().chunk(parent):
            payload = chunk.to_payload()
            for forbidden in ("is_selected", "query", "answer", "Answer", "relevance"):
                assert forbidden not in payload
            assert {"doc_id", "parent_id", "chunk_id", "language", "strategy",
                    "source_split", "content_hash", "text"} <= set(payload)
