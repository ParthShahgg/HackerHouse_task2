"""Normalisation, content hashing and multilingual sentence splitting."""

from __future__ import annotations

import unicodedata

import pytest

from app.indexing.normalize import (
    approx_token_count,
    content_hash,
    make_doc_id,
    normalize_query,
    normalize_text,
    split_sentences,
    strip_asr_artifacts,
)


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("  a   b \t\n c  ") == "a b c"

    def test_empty_and_none(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""

    def test_is_idempotent(self):
        """Required: the hash is derived from this, so it must be stable."""
        raw = "  निगम \u200b एक\tकंपनी  \n है।  "
        once = normalize_text(raw)
        assert normalize_text(once) == once

    def test_applies_nfc(self):
        # Devanagari KA + NUKTA (decomposed) must fold to the precomposed form,
        # otherwise identical-looking passages hash differently.
        decomposed = "\u0915\u093c"
        result = normalize_text(decomposed)
        assert result == unicodedata.normalize("NFC", decomposed)

    def test_strips_zero_width_space_and_bom(self):
        assert normalize_text("a\u200bb\ufeffc") == "abc"

    def test_strips_control_characters(self):
        assert normalize_text("a\x00\x07b") == "a b"

    def test_preserves_devanagari_and_danda(self):
        text = "निगम एक कंपनी है।"
        assert normalize_text(text) == text

    def test_does_not_lowercase(self):
        """Case folding would damage English named entities."""
        assert normalize_text("Manhattan Project") == "Manhattan Project"

    def test_newlines_become_spaces(self):
        assert normalize_text("line1\nline2\r\nline3") == "line1 line2 line3"


class TestContentHash:
    def test_deterministic(self, hindi_passage):
        assert content_hash("hi", hindi_passage) == content_hash("hi", hindi_passage)

    def test_language_scoped(self, hindi_passage):
        """Same text in two languages must be two documents."""
        assert content_hash("hi", hindi_passage) != content_hash("mr", hindi_passage)

    def test_whitespace_insensitive(self):
        assert content_hash("hi", "a b") == content_hash("hi", "  a   b  ")

    def test_no_boundary_collision(self):
        """('hi','xy') must not collide with ('hix','y')."""
        assert content_hash("hi", "xy") != content_hash("hix", "y")

    def test_ignores_invisible_joiners(self):
        """ZWNJ/ZWJ differences must not defeat dedup."""
        assert content_hash("hi", "क\u200cख") == content_hash("hi", "कख")

    def test_different_text_differs(self):
        assert content_hash("hi", "abc") != content_hash("hi", "abd")

    def test_hex_sha256_length(self):
        assert len(content_hash("hi", "abc")) == 64


class TestDocId:
    def test_format(self):
        chash = content_hash("hi", "text")
        doc_id = make_doc_id("hi", chash)
        assert doc_id == f"hi:{chash[:16]}"
        assert doc_id.startswith("hi:")

    def test_stable(self):
        chash = content_hash("ta", "text")
        assert make_doc_id("ta", chash) == make_doc_id("ta", chash)


class TestSplitSentences:
    def test_devanagari_danda(self):
        text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
        assert len(split_sentences(text)) == 3

    def test_english_full_stop(self):
        assert len(split_sentences("One. Two. Three.")) == 3

    def test_tamil_uses_full_stop(self):
        text = "இது முதல் வாக்கியம். இது இரண்டாவது வாக்கியம்."
        assert len(split_sentences(text)) == 2

    def test_urdu_full_stop(self):
        assert len(split_sentences("پہلا جملہ۔ دوسرا جملہ۔")) == 2

    def test_question_and_exclamation(self):
        assert len(split_sentences("What? Really! Yes.")) == 3

    def test_no_terminator_is_one_sentence(self):
        assert split_sentences("no terminator here") == ["no terminator here"]

    def test_empty(self):
        assert split_sentences("") == []

    def test_abbreviation_not_a_boundary(self):
        """'Dr. Smith' is one sentence, not two."""
        assert len(split_sentences("Dr. Smith works here. He is nice.")) == 2

    def test_decimal_not_a_boundary(self):
        assert len(split_sentences("It costs 3.5 dollars today. That is fine.")) == 2

    def test_initials_not_a_boundary(self):
        assert len(split_sentences("J. Smith arrived late. We waited.")) == 2

    def test_no_empty_fragments(self):
        for sentence in split_sentences("A।  ।  B।"):
            assert sentence.strip()

    def test_reassembly_preserves_content(self, hindi_passage):
        sentences = split_sentences(hindi_passage)
        rejoined = " ".join(sentences)
        assert set(rejoined.split()) == set(normalize_text(hindi_passage).split())


class TestApproxTokenCount:
    def test_zero_for_empty(self):
        assert approx_token_count("") == 0

    def test_monotonic_in_length(self):
        short = approx_token_count("one two three")
        long = approx_token_count("one two three " * 30)
        assert long > short

    def test_devanagari_nonzero(self):
        assert approx_token_count("निगम एक कंपनी है") > 0


class TestASRArtifacts:
    def test_removes_fillers(self):
        cleaned, removed = strip_asr_artifacts("um what is uh a corporation")
        assert "um" not in cleaned.split()
        assert "uh" not in cleaned.split()
        assert "fillers" in removed

    def test_removes_bracketed_annotations(self):
        cleaned, removed = strip_asr_artifacts("what is [inaudible] a corporation")
        assert "inaudible" not in cleaned
        assert "annotations" in removed

    def test_collapses_stutter(self):
        cleaned, removed = strip_asr_artifacts("what what is a corporation")
        assert cleaned.lower().count("what") == 1
        assert "stutter" in removed

    def test_collapses_repeated_punctuation(self):
        cleaned, removed = strip_asr_artifacts("what is this???")
        assert "???" not in cleaned
        assert "repeated_punct" in removed

    def test_preserves_named_entities(self):
        """Aggressive rewriting would destroy exactly what retrieval keys on."""
        cleaned, _ = strip_asr_artifacts("um tell me about the Manhattan Project")
        assert "Manhattan Project" in cleaned

    def test_never_empties_the_query(self):
        cleaned, _ = strip_asr_artifacts("um")
        assert cleaned.strip()

    def test_preserves_devanagari(self):
        query = "कॉर्पोरेशन क्या है?"
        cleaned, _ = strip_asr_artifacts(query)
        assert "कॉर्पोरेशन" in cleaned


class TestNormalizeQuery:
    def test_returns_text_and_artifacts(self):
        cleaned, removed = normalize_query("  um   what  is  a  corporation?  ")
        assert cleaned
        assert isinstance(removed, list)

    def test_preserves_script(self):
        cleaned, _ = normalize_query("कॉर्पोरेशन क्या है?")
        assert any("\u0900" <= ch <= "\u097f" for ch in cleaned)

    @pytest.mark.parametrize(
        "query",
        [
            "कॉर्पोरेशन क्या है?",
            "महामंडळ काय आहे?",
            "நிறுவனம் என்றால் என்ன?",
            "కార్పొరేషన్ అంటే ఏమిటి?",
            "What is a corporation?",
        ],
    )
    def test_multilingual_queries_survive(self, query):
        cleaned, _ = normalize_query(query)
        assert cleaned.strip()
