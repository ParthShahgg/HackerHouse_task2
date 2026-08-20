"""Passage deduplication and the dataset-leakage invariants."""

from __future__ import annotations

from app.indexing.deduplicate import PassageDeduplicator
from app.indexing.normalize import content_hash
from app.indexing.records import Chunk, DatasetRow


class TestDeduplicator:
    def test_first_add_is_new(self):
        dedup = PassageDeduplicator()
        chash, is_new = dedup.add("some passage", language="hi", source_split="validation")
        assert is_new and chash
        assert len(dedup) == 1

    def test_repeat_is_deduplicated(self):
        """The core requirement: one passage across many query rows -> one doc."""
        dedup = PassageDeduplicator()
        for query_id in range(10):
            dedup.add("same passage", language="hi", source_split="validation", query_id=query_id)
        assert dedup.unique_count == 1
        assert dedup.duplicates_removed == 9
        assert dedup.passages_seen == 10

    def test_whitespace_variants_deduplicate(self):
        dedup = PassageDeduplicator()
        dedup.add("a b", language="hi", source_split="v")
        dedup.add("  a   b  ", language="hi", source_split="v")
        assert dedup.unique_count == 1

    def test_language_scoped(self):
        dedup = PassageDeduplicator()
        dedup.add("same text", language="hi", source_split="v")
        dedup.add("same text", language="mr", source_split="v")
        assert dedup.unique_count == 2

    def test_empty_passages_skipped(self):
        dedup = PassageDeduplicator()
        assert dedup.add("", language="hi", source_split="v") == (None, False)
        assert dedup.add("   ", language="hi", source_split="v") == (None, False)
        assert dedup.empty_skipped == 2
        assert dedup.unique_count == 0

    def test_provenance_retained(self):
        dedup = PassageDeduplicator()
        for query_id in (11, 22, 33):
            dedup.add("shared", language="hi", source_split="v", query_id=query_id)
        parent = dedup.parents[0]
        assert set(parent.source_query_ids) == {11, 22, 33}

    def test_provenance_is_capped(self):
        dedup = PassageDeduplicator(max_provenance=4)
        for query_id in range(50):
            dedup.add("shared", language="hi", source_split="v", query_id=query_id)
        assert len(dedup.parents[0].source_query_ids) <= 4

    def test_add_many_returns_hashes(self):
        dedup = PassageDeduplicator()
        hashes = dedup.add_many(["a", "b", "", "a"], language="hi", source_split="v")
        assert len(hashes) == 3          # empty skipped
        assert len(set(hashes)) == 2     # "a" repeated

    def test_dedup_ratio(self):
        dedup = PassageDeduplicator()
        dedup.add_many(["a", "a", "a", "b"], language="hi", source_split="v")
        assert dedup.dedup_ratio == 0.5

    def test_doc_id_derived_from_hash(self):
        dedup = PassageDeduplicator()
        chash, _ = dedup.add("text", language="ta", source_split="v")
        parent = dedup.get(chash)
        assert parent.doc_id == f"ta:{chash[:16]}"

    def test_membership(self):
        dedup = PassageDeduplicator()
        chash, _ = dedup.add("text", language="hi", source_split="v")
        assert chash in dedup
        assert "nope" not in dedup


class TestLeakagePrevention:
    """Structural guarantees that labels/answers cannot enter the index."""

    def test_chunk_payload_has_no_label_fields(self, sample_chunk: Chunk):
        payload = sample_chunk.to_payload()
        for forbidden in ("is_selected", "query", "answer", "Answer", "Eng_Answer", "query_type"):
            assert forbidden not in payload

    def test_chunk_dataclass_has_no_label_fields(self):
        fields = set(Chunk.__dataclass_fields__)
        for forbidden in ("is_selected", "query", "answer"):
            assert forbidden not in fields

    def test_retrieval_candidate_has_no_label_fields(self):
        from app.schemas.retrieval import RetrievalCandidate

        fields = set(RetrievalCandidate.model_fields)
        for forbidden in ("is_selected", "relevance", "answer", "label"):
            assert forbidden not in fields

    def test_indexed_text_is_passage_only(self):
        """The indexed document must be the passage - never query+answer+passage."""
        from app.chunking import ChunkingEngine
        from app.indexing.normalize import make_doc_id, normalize_text
        from app.indexing.records import ParentPassage

        row = DatasetRow(
            query_id=1,
            language="hi",
            split="validation",
            query="कॉर्पोरेशन क्या है?",
            answer="निगम एक कंपनी है जो एकल इकाई के रूप में कार्य करती है।",
            query_type="DESCRIPTION",
            passages=["निगम एक कंपनी या लोगों का समूह होता है। यह कानून द्वारा मान्यता प्राप्त है। तीसरा वाक्य।"],
            is_selected=[1],
        )
        text = normalize_text(row.passages[0])
        chash = content_hash(row.language, text)
        parent = ParentPassage(
            doc_id=make_doc_id(row.language, chash),
            content_hash=chash,
            language=row.language,
            text=text,
            source_split=row.split,
        )
        for chunk in ChunkingEngine().chunk(parent):
            assert row.query not in chunk.text
            assert row.answer not in chunk.text
            # Every chunk is a span of the source passage.
            assert chunk.text.replace(" ", "") in text.replace(" ", "") or chunk.text == text

    def test_eval_example_keeps_labels_separately(self):
        from app.indexing.records import EvalExample

        example = EvalExample(
            query_id=1, language="hi", split="validation",
            query="q", answer="a", query_type="DESCRIPTION",
            relevant_hashes=["h1"], candidate_hashes=["h1", "h2"],
        )
        assert example.has_label
        # Labels live here, in the evaluation layer - not in Chunk.
        assert "relevant_hashes" in example.to_json()

    def test_eval_split_is_deterministic_and_disjoint(self):
        from app.indexing.corpus import eval_split_of

        assignments = {qid: eval_split_of(qid) for qid in range(500)}
        # stable across calls
        assert all(eval_split_of(qid) == split for qid, split in assignments.items())
        splits = set(assignments.values())
        assert splits <= {"calibration", "test"}
        # both populated, roughly at the configured fraction
        calibration = sum(1 for s in assignments.values() if s == "calibration")
        assert 0.25 < calibration / len(assignments) < 0.55
