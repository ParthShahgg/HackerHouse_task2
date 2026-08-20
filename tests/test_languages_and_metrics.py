"""Language code mapping (a real source of silent bugs) and ranking metrics."""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    collapse_to_unique,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.languages import (
    LANGUAGES,
    TRAIN_LANGUAGES,
    VALIDATION_LANGUAGES,
    dataset_filename,
    has_split,
    iso1_to_sarvam,
    normalize_language,
    sarvam_to_iso1,
    script_of,
)
from app.observability.metrics import MetricsRegistry, percentile
from app.schemas.common import LatencyBreakdown


class TestLanguageMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("hi", "hi"), ("hi-IN", "hi"), ("HI-in", "hi"), ("hin", "hi"),
            ("hin_Deva", "hi"), ("mr-IN", "mr"), ("ta-IN", "ta"), ("te-IN", "te"),
            ("od-IN", "or"), ("or-IN", "or"), ("ori", "or"), ("pan", "pa"),
            ("tel", "te"), ("en-IN", "en"), ("  hi  ", "hi"),
        ],
    )
    def test_normalise(self, raw, expected):
        assert normalize_language(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "xx", "klingon", "   "])
    def test_unknown_returns_none(self, raw):
        """None lets the caller go cross-lingual instead of guessing."""
        assert normalize_language(raw) is None

    def test_sarvam_round_trip(self):
        for iso1 in ("hi", "mr", "ta", "te", "bn", "ur"):
            assert sarvam_to_iso1(iso1_to_sarvam(iso1)) == iso1

    def test_file_codes_are_not_truncations(self):
        """The trap this table exists to avoid."""
        assert LANGUAGES["or"].file_code == "ori"
        assert LANGUAGES["pa"].file_code == "pan"
        assert LANGUAGES["te"].file_code == "tel"
        assert LANGUAGES["as"].file_code == "asm"

    @pytest.mark.parametrize(
        ("iso1", "split", "expected"),
        [
            ("hi", "train", "train/hintrain.parquet"),
            ("hi", "validation", "validation/hinval.parquet"),
            ("mr", "train", "train/martrain.parquet"),
            ("ta", "validation", "validation/tamval.parquet"),
            ("te", "validation", "validation/telval.parquet"),
            ("or", "train", "train/oritrain.parquet"),
        ],
    )
    def test_dataset_filename(self, iso1, split, expected):
        assert dataset_filename(iso1, split) == expected

    def test_telugu_train_is_absent_upstream(self):
        """Real gap: the README advertises teltrain.jsonl but it does not exist."""
        assert not has_split("te", "train")
        assert has_split("te", "validation")
        with pytest.raises(FileNotFoundError, match="no 'train' file"):
            dataset_filename("te", "train")

    def test_train_languages_exclude_telugu(self):
        assert "te" not in TRAIN_LANGUAGES
        assert "te" in VALIDATION_LANGUAGES
        assert len(TRAIN_LANGUAGES) == 13

    def test_english_has_no_shard(self):
        """English is a derived representation, not a dataset file."""
        assert not has_split("en", "train")
        assert not has_split("en", "validation")

    def test_unsupported_language_raises(self):
        with pytest.raises(KeyError):
            dataset_filename("klingon", "train")

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError):
            dataset_filename("hi", "test")

    @pytest.mark.parametrize(
        ("text", "script"),
        [
            ("निगम एक कंपनी", "Deva"),
            ("நிறுவனம் என்றால்", "Taml"),
            ("కార్పొరేషన్ అంటే", "Telu"),
            ("A corporation is", "Latn"),
            ("کمپنی کیا ہے", "Arab"),
        ],
    )
    def test_script_detection(self, text, script):
        assert script_of(text) == script

    def test_script_of_punctuation_only(self):
        assert script_of("... !!! 123") is None


class TestRankingMetrics:
    def test_collapse_keeps_best_rank(self):
        assert collapse_to_unique(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_collapse_drops_empty(self):
        assert collapse_to_unique(["", "a", "", "b"]) == ["a", "b"]

    def test_recall_denominator_is_all_relevant(self):
        """1 of 2 relevant found at rank 1 -> 0.5, not 1.0."""
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 1) == 0.5
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0

    def test_recall_zero_when_missed(self):
        assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_recall_nan_without_labels(self):
        assert math.isnan(recall_at_k(["a"], set(), 1))

    def test_precision(self):
        assert precision_at_k(["a", "x"], {"a"}, 2) == 0.5
        assert precision_at_k([], {"a"}, 2) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_perfect_and_missed(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
        assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0

    def test_ndcg_rewards_higher_rank(self):
        assert ndcg_at_k(["a", "x"], {"a"}, 10) > ndcg_at_k(["x", "a"], {"a"}, 10)

    def test_ndcg_bounded(self):
        assert 0.0 <= ndcg_at_k(["x", "a", "y"], {"a", "b"}, 10) <= 1.0

    def test_evaluate_ranking_shape(self):
        result = evaluate_ranking(["a", "b", "c"], {"a"})
        for key in ("recall@1", "recall@5", "mrr", "ndcg@10"):
            assert key in result

    def test_fragmentation_confers_no_advantage(self):
        """Core fairness property of the chunking comparison."""
        fragmented = ["p1", "p1", "p1", "p2"]
        compact = ["p1", "p2"]
        assert recall_at_k(collapse_to_unique(fragmented), {"p2"}, 2) == recall_at_k(
            collapse_to_unique(compact), {"p2"}, 2
        )


class TestPercentile:
    def test_edges(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 5.0

    def test_p100_is_observed_max(self):
        assert percentile([1.0, 99.0], 100) == 99.0

    def test_p50(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0

    def test_empty_is_nan(self):
        assert math.isnan(percentile([], 50))

    def test_single(self):
        assert percentile([7.0], 70) == 7.0


class TestMetricsRegistry:
    def test_counts_requests_and_abstentions(self):
        registry = MetricsRegistry()
        registry.record_request(abstained=False, grounded=True)
        registry.record_request(abstained=True, grounded=False, abstain_reason="low_confidence")
        snapshot = registry.snapshot()
        assert snapshot["requests_total"] == 2
        assert snapshot["abstentions_total"] == 1
        assert snapshot["abstention_rate"] == 0.5
        assert snapshot["abstain_reasons"]["low_confidence"] == 1

    def test_unmeasured_stages_not_recorded_as_zero(self):
        """Otherwise percentiles get polluted with fake zeros."""
        registry = MetricsRegistry()
        registry.record_latency(LatencyBreakdown(rerank_latency=10.0, stt_latency=None))
        stages = registry.snapshot()["by_stage_latency_ms"]
        assert "rerank_latency" in stages
        assert "stt_latency" not in stages

    def test_percentiles_reported(self):
        registry = MetricsRegistry()
        for value in range(1, 11):
            registry.observe("rerank_latency", float(value))
        stats = registry.snapshot()["by_stage_latency_ms"]["rerank_latency"]
        assert stats["count"] == 10
        assert stats["p100"] == 10.0

    def test_reset(self):
        registry = MetricsRegistry()
        registry.record_request(abstained=False, grounded=True)
        registry.reset()
        assert registry.snapshot()["requests_total"] == 0

    def test_thread_safe_under_concurrency(self):
        import threading

        registry = MetricsRegistry()

        def worker():
            for _ in range(200):
                registry.incr("hits")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert registry.snapshot()["counters"]["hits"] == 1600
