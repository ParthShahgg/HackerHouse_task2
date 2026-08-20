"""Unit tests for RRF fusion, language routing, parent expansion, confidence gate."""

from __future__ import annotations

import pytest

from app.retrieval.confidence import ConfidenceGate, GateThresholds, abstain_message
from app.retrieval.hybrid import BranchHit, decide_languages, reciprocal_rank_fusion
from app.retrieval.parent_expansion import expand_to_parents
from app.schemas.common import AbstainReason, GateDecision, RetrievalMode
from _factories import make_reranked


def hit(chunk_id: str, rank: int, score: float = 0.5) -> BranchHit:
    return BranchHit(
        chunk_id=chunk_id,
        payload={"chunk_id": chunk_id, "parent_id": chunk_id, "text": "t", "language": "hi"},
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
class TestRRF:
    def test_empty(self):
        assert reciprocal_rank_fusion({}) == []

    def test_single_branch_preserves_order(self):
        branch = [hit("a", 1), hit("b", 2), hit("c", 3)]
        fused = reciprocal_rank_fusion({"dense": branch})
        assert [cid for cid, _, _ in fused] == ["a", "b", "c"]

    def test_known_scores(self):
        """RRF(d) = sum 1/(k + rank); verify arithmetic exactly."""
        fused = reciprocal_rank_fusion(
            {"dense": [hit("a", 1)], "sparse": [hit("a", 2)]}, k=60
        )
        assert fused[0][0] == "a"
        assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)

    def test_document_in_both_branches_outranks_single_branch(self):
        """The property that makes RRF useful: agreement wins."""
        fused = reciprocal_rank_fusion(
            {
                "dense": [hit("both", 2), hit("dense_only", 1)],
                "sparse": [hit("both", 2), hit("sparse_only", 1)],
            },
            k=60,
        )
        assert fused[0][0] == "both"

    def test_ignores_score_magnitude(self):
        """Only ranks matter - this is why no calibration is needed."""
        low = reciprocal_rank_fusion({"dense": [hit("a", 1, score=0.01)]}, k=60)
        high = reciprocal_rank_fusion({"dense": [hit("a", 1, score=999.0)]}, k=60)
        assert low[0][1] == high[0][1]

    def test_limit_applied(self):
        branch = [hit(f"c{i}", i) for i in range(1, 11)]
        assert len(reciprocal_rank_fusion({"dense": branch}, limit=3)) == 3

    def test_deterministic_tie_break(self):
        """Reproducible ordering, so benchmark percentiles don't jitter."""
        branches = {"dense": [hit("b", 1), hit("a", 1)]}
        first = [cid for cid, _, _ in reciprocal_rank_fusion(branches)]
        second = [cid for cid, _, _ in reciprocal_rank_fusion(branches)]
        assert first == second

    def test_provenance_recorded(self):
        fused = reciprocal_rank_fusion({"dense": [hit("a", 1)], "sparse": [hit("a", 3)]})
        _cid, _score, provenance = fused[0]
        assert set(provenance) == {"dense", "sparse"}

    def test_larger_k_flattens_ranking(self):
        branches = {"dense": [hit("a", 1), hit("b", 2)]}
        small = reciprocal_rank_fusion(branches, k=1)
        large = reciprocal_rank_fusion(branches, k=1000)
        assert (small[0][1] - small[1][1]) > (large[0][1] - large[1][1])


# ---------------------------------------------------------------------------
# Language-aware retrieval
# ---------------------------------------------------------------------------
class TestDecideLanguages:
    POOL = ["hi", "mr", "ta", "te"]

    def test_confident_detection_filters(self):
        langs, mode = decide_languages(
            detected_language="hi", confidence=0.95, is_code_mixed=False,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == ["hi"]
        assert mode == RetrievalMode.LANGUAGE_FILTERED

    def test_low_confidence_goes_cross_lingual(self):
        langs, mode = decide_languages(
            detected_language="hi", confidence=0.3, is_code_mixed=False,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == self.POOL
        assert mode == RetrievalMode.CROSS_LINGUAL

    def test_code_mixed_never_filters(self):
        """Filtering a Hindi-English utterance can make the answer unreachable."""
        langs, mode = decide_languages(
            detected_language="hi", confidence=0.99, is_code_mixed=True,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == self.POOL
        assert mode == RetrievalMode.CODE_MIXED_CROSS_LINGUAL

    def test_unknown_language_goes_cross_lingual(self):
        langs, mode = decide_languages(
            detected_language=None, confidence=None, is_code_mixed=False,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == self.POOL
        assert mode == RetrievalMode.CROSS_LINGUAL

    def test_unindexed_language_goes_cross_lingual(self):
        langs, mode = decide_languages(
            detected_language="bn", confidence=0.99, is_code_mixed=False,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == self.POOL
        assert mode == RetrievalMode.CROSS_LINGUAL

    def test_missing_confidence_does_not_filter(self):
        langs, _ = decide_languages(
            detected_language="hi", confidence=None, is_code_mixed=False,
            configured=self.POOL, min_confidence=0.65,
        )
        assert langs == self.POOL


# ---------------------------------------------------------------------------
# Parent expansion
# ---------------------------------------------------------------------------
class TestParentExpansion:
    def test_empty(self):
        assert expand_to_parents([], limit=5) == []

    def test_collapses_children_of_same_parent(self):
        """The documented example: 4 children -> 3 parents."""
        candidates = [
            make_reranked("child_17", 5.0, parent_id="parent_A"),
            make_reranked("child_22", 4.0, parent_id="parent_A"),
            make_reranked("child_51", 3.0, parent_id="parent_B"),
            make_reranked("child_73", 2.0, parent_id="parent_C"),
        ]
        contexts = expand_to_parents(candidates, limit=5)
        assert [c.parent_id for c in contexts] == ["parent_A", "parent_B", "parent_C"]

    def test_parent_score_is_best_child_not_sum(self):
        """Otherwise a fragmented passage climbs by having many windows."""
        candidates = [
            make_reranked("c1", 5.0, parent_id="A"),
            make_reranked("c2", 4.0, parent_id="A"),
            make_reranked("c3", 4.5, parent_id="B"),
        ]
        contexts = expand_to_parents(candidates, limit=5)
        by_id = {c.parent_id: c for c in contexts}
        assert by_id["A"].best_score == 5.0
        assert by_id["B"].best_score == 4.5

    def test_supporting_chunk_ids_recorded(self):
        candidates = [
            make_reranked("c1", 5.0, parent_id="A"),
            make_reranked("c2", 4.0, parent_id="A"),
        ]
        contexts = expand_to_parents(candidates, limit=5)
        assert contexts[0].supporting_chunk_ids == ["c1", "c2"]

    def test_limit_applied_to_parents(self):
        candidates = [make_reranked(f"c{i}", 5.0 - i, parent_id=f"p{i}") for i in range(10)]
        assert len(expand_to_parents(candidates, limit=3)) == 3

    def test_rank_order_preserved(self):
        candidates = [
            make_reranked("c1", 9.0, parent_id="high"),
            make_reranked("c2", 1.0, parent_id="low"),
        ]
        contexts = expand_to_parents(candidates, limit=5)
        assert contexts[0].parent_id == "high"

    def test_parent_lookup_replaces_child_text(self):
        candidates = [
            make_reranked("c1", 5.0, parent_id="A", text="short window", strategy="sentence_window")
        ]
        full = "the full parent passage which is considerably longer than the window"
        contexts = expand_to_parents(
            candidates, limit=5, parent_text_lookup=lambda pid: full
        )
        assert contexts[0].text == full

    def test_native_text_preferred_over_window(self):
        candidates = [
            make_reranked("w", 5.0, parent_id="A", text="win", strategy="sentence_window"),
            make_reranked("A", 4.0, parent_id="A", text="the full native passage", strategy="native"),
        ]
        contexts = expand_to_parents(candidates, limit=5)
        assert contexts[0].text == "the full native passage"

    def test_lookup_failure_is_tolerated(self):
        def boom(pid):
            raise RuntimeError("qdrant down")

        candidates = [make_reranked("c1", 5.0, parent_id="A", text="child text",
                                    strategy="sentence_window")]
        contexts = expand_to_parents(candidates, limit=5, parent_text_lookup=boom)
        assert contexts[0].text == "child text"

    def test_disabled_keeps_children_separate(self):
        candidates = [
            make_reranked("c1", 5.0, parent_id="A"),
            make_reranked("c2", 4.0, parent_id="A"),
        ]
        contexts = expand_to_parents(candidates, limit=5, enabled=False)
        assert len(contexts) == 2
        assert contexts[0].citation_id == "c1"

    def test_citation_id_is_parent_when_enabled(self):
        candidates = [make_reranked("c1", 5.0, parent_id="A")]
        assert expand_to_parents(candidates, limit=5)[0].citation_id == "A"


# ---------------------------------------------------------------------------
# Confidence / abstention gate
# ---------------------------------------------------------------------------
class TestConfidenceGate:
    def test_no_candidates_abstains(self, rerank_result_factory):
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=0.0, calibrated=True)
        ).evaluate(rerank_result_factory([]))
        assert decision.decision == GateDecision.ABSTAIN
        assert decision.reason == AbstainReason.NO_CANDIDATES

    def test_high_score_generates(self, rerank_result_factory):
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=0.0, calibrated=True)
        ).evaluate(rerank_result_factory([5.0, 1.0]))
        assert decision.decision == GateDecision.GENERATE

    def test_low_score_abstains(self, rerank_result_factory):
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=2.0, rerank_margin_min=0.0, calibrated=True)
        ).evaluate(rerank_result_factory([1.0, 0.5]))
        assert decision.decision == GateDecision.ABSTAIN
        assert decision.reason == AbstainReason.LOW_CONFIDENCE

    def test_weak_margin_abstains(self, rerank_result_factory):
        """Retrieval ambiguity: several passages look equally plausible."""
        decision = ConfidenceGate(
            GateThresholds(
                rerank_abstain_below=0.0, rerank_margin_min=1.0,
                margin_override_score=100.0, calibrated=True,
            )
        ).evaluate(rerank_result_factory([2.0, 1.95]))
        assert decision.decision == GateDecision.ABSTAIN
        assert decision.reason == AbstainReason.WEAK_MARGIN

    def test_decisive_top_score_overrides_weak_margin(self, rerank_result_factory):
        """Two passages that both genuinely answer must not be punished."""
        decision = ConfidenceGate(
            GateThresholds(
                rerank_abstain_below=0.0, rerank_margin_min=1.0,
                margin_override_score=6.0, calibrated=True,
            )
        ).evaluate(rerank_result_factory([9.0, 8.9]))
        assert decision.decision == GateDecision.GENERATE

    def test_single_candidate_has_no_margin(self, rerank_result_factory):
        result = rerank_result_factory([5.0])
        assert result.margin is None
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=1.0, calibrated=True)
        ).evaluate(result)
        assert decision.decision == GateDecision.GENERATE

    def test_calibration_flag_propagates(self, rerank_result_factory):
        uncalibrated = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=0.0, calibrated=False)
        ).evaluate(rerank_result_factory([5.0, 1.0]))
        assert uncalibrated.thresholds_calibrated is False

        calibrated = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=0.0, calibrated=True)
        ).evaluate(rerank_result_factory([5.0, 1.0]))
        assert calibrated.thresholds_calibrated is True

    def test_thresholds_reported_in_decision(self, rerank_result_factory):
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=1.5, rerank_margin_min=0.25, calibrated=True)
        ).evaluate(rerank_result_factory([5.0, 1.0]))
        assert decision.threshold_used == 1.5
        assert decision.margin_threshold_used == 0.25

    def test_margin_check_disabled_at_zero(self, rerank_result_factory):
        decision = ConfidenceGate(
            GateThresholds(rerank_abstain_below=0.0, rerank_margin_min=0.0, calibrated=True)
        ).evaluate(rerank_result_factory([1.0, 0.999]))
        assert decision.decision == GateDecision.GENERATE


class TestAbstainMessage:
    def test_english_default(self):
        assert "enough information" in abstain_message(None).lower()

    @pytest.mark.parametrize("language", ["hi", "mr", "ta", "te", "bn", "ur"])
    def test_localised(self, language):
        message = abstain_message(language)
        assert message
        assert message != abstain_message("en")

    def test_unknown_language_falls_back(self):
        assert abstain_message("xx") == abstain_message("en")
