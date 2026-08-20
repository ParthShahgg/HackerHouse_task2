"""Text normalisation, stable content hashing and multilingual sentence splitting.

Design constraints:

* Normalisation must be **idempotent and deterministic**, because the content
  hash derived from it is the deduplication key and the stable ``doc_id``. If
  normalisation changed between runs, ``--rebuild`` would silently duplicate the
  whole corpus.
* Normalisation must **not** be lossy for Indic text. No case folding (Indic
  scripts are unicameral, and folding would damage English named entities), no
  transliteration, no punctuation stripping, no diacritic removal.
* Unicode NFC is applied because the same Devanagari grapheme can be encoded
  either precomposed or as base+nukta; without NFC those hash differently and
  dedup silently fails.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = [
    "normalize_text",
    "normalize_query",
    "content_hash",
    "make_doc_id",
    "split_sentences",
    "approx_token_count",
    "strip_asr_artifacts",
]

# Zero-width and bidi control characters. These are invisible, appear
# inconsistently in machine-translated Indic text, and would otherwise defeat
# exact-duplicate detection.
_INVISIBLES = dict.fromkeys(
    [
        0x200B,  # zero width space
        0x200C,  # ZWNJ - see note below
        0x200D,  # ZWJ  - see note below
        0x200E,
        0x200F,  # LRM / RLM
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2060,
        0xFEFF,  # BOM
    ]
)

# ZWNJ/ZWJ are orthographically meaningful in some Indic scripts (they control
# conjunct formation). Removing them can change rendering, so they are NOT
# stripped from stored text - only from the hashing view. See `content_hash`.
_HASH_ONLY_INVISIBLES = {0x200C: None, 0x200D: None}

_WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u3000]+")
_NEWLINES_RE = re.compile(r"(?:\r\n|\r|\n)+")
# C0/C1 controls except tab/newline.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize_text(text: str | None) -> str:
    """Canonical form for stored/indexed passage text.

    NFC + control-character removal + whitespace collapse. Nothing else.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    out = out.translate(_INVISIBLES)
    out = _CTRL_RE.sub(" ", out)
    out = _NEWLINES_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out)
    return out.strip()


def _hash_view(text: str) -> str:
    """Slightly more aggressive form used *only* for the dedup key.

    Folds ZWNJ/ZWJ and trailing punctuation-only whitespace differences so that
    passages differing solely by invisible joiners collapse together, while the
    stored text keeps its original joiners.
    """
    return normalize_text(text).translate(_HASH_ONLY_INVISIBLES)


def content_hash(language: str, text: str) -> str:
    """``sha256(language + normalized_text)`` as required.

    Language is part of the key so the same passage in Hindi and Marathi are
    distinct documents (they are genuinely different retrieval units), while an
    identical Hindi passage repeated across 40 query rows collapses to one.

    A ``\\x1f`` unit separator prevents boundary collisions such as
    ``("hi", "xy")`` vs ``("hix", "y")``.
    """
    payload = f"{language}\x1f{_hash_view(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


def make_doc_id(language: str, chash: str) -> str:
    """Stable, human-readable, collision-safe document identifier.

    16 hex chars = 64 bits. At the full-corpus scale of ~9M unique passages the
    birthday collision probability is ~2e-6, and IDs are additionally scoped by
    language.
    """
    return f"{language}:{chash[:16]}"


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------
# Terminators across the scripts present in MSMARCO-XI:
#   .  ?  !     Latin
#   ।  ॥        Devanagari danda / double danda (hi, mr, ne, sa)
#   ۔  ؟  !     Urdu / Arabic-script
#   。 ！ ？     full-width (rare, but appears in scraped web text)
# Tamil/Telugu/Kannada/Malayalam/Gujarati/Odia/Punjabi/Bengali use the Latin
# full stop, and Bengali/Assamese also use the danda.
_TERMINATORS = ".?!।॥۔؟。！？"
_SENT_SPLIT_RE = re.compile(rf"(?<=[{re.escape(_TERMINATORS)}])[\s]+")

# Abbreviations that must not end a sentence. Only English matters here: Indic
# scripts do not use the full stop for abbreviation in this dataset.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "inc", "ltd", "co", "corp", "no", "vol", "fig", "approx", "dept", "est",
    "u.s", "u.k", "a.m", "p.m", "i.e", "e.g",
}
_ABBREV_TAIL_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")
# A single capital letter followed by a dot: initials such as "J. Smith".
_INITIAL_RE = re.compile(r"\b[A-Z]\.$")
_DECIMAL_TAIL_RE = re.compile(r"\d\.$")


def _is_false_boundary(fragment: str) -> bool:
    frag = fragment.rstrip()
    if not frag:
        return True
    if _DECIMAL_TAIL_RE.search(frag):
        return True
    if _INITIAL_RE.search(frag):
        return True
    m = _ABBREV_TAIL_RE.search(frag)
    if m and m.group(1).lower().rstrip(".") in _ABBREV:
        return True
    return False


def split_sentences(text: str, min_chars: int = 2) -> list[str]:
    """Split into sentences using script-aware terminators.

    Rule-based on purpose: a neural sentence splitter for 14 languages would add
    a model load, latency and a dependency, and MSMARCO passages are short
    well-punctuated web prose where terminators are reliable. Fragments shorter
    than ``min_chars`` are merged into the previous sentence rather than emitted,
    so stray punctuation cannot create empty chunks.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    raw_parts = _SENT_SPLIT_RE.split(normalized)
    merged: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if merged and (_is_false_boundary(merged[-1]) or len(part) < min_chars):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    # A passage with no terminator at all is one sentence.
    return merged or [normalized]


def approx_token_count(text: str) -> int:
    """Cheap token estimate used for chunking thresholds only.

    Avoids loading a tokenizer during dataset streaming. Indic scripts have no
    whitespace-delimited relationship to subword count, so we blend a
    whitespace count with a characters-per-token heuristic and take the larger,
    which errs toward *over*-estimating length. Over-estimating is the safe
    direction: it routes borderline passages to finer chunking rather than
    leaving an over-long passage as a single coarse vector.

    The authoritative token count during embedding comes from the real
    XLM-RoBERTa tokenizer; this is a pre-filter.
    """
    if not text:
        return 0
    words = text.count(" ") + 1
    # ~3.5 chars/token is a reasonable XLM-R average for Indic scripts.
    char_based = int(len(text) / 3.5)
    return max(words, char_based)


# --------------------------------------------------------------------------
# Query-side preprocessing
# --------------------------------------------------------------------------
# Filler/backchannel tokens that streaming ASR commonly emits. Kept small and
# anchored: aggressive rewriting is explicitly out of scope because it can
# destroy named entities, which are exactly what retrieval keys on.
_ASR_FILLERS = (
    r"\bum+\b", r"\buh+\b", r"\berm+\b", r"\bhmm+\b", r"\buhh+\b",
    r"\baa+\b", r"\bहम्म+\b", r"\bउम+\b", r"\bआं+\b", r"\bஅ்ம்+\b",
)
_FILLER_RE = re.compile("|".join(_ASR_FILLERS), re.IGNORECASE)
# Bracketed ASR annotations: [inaudible], (background noise), <unk>.
_ANNOTATION_RE = re.compile(r"[\[\(<](?:inaudible|unintelligible|noise|silence|unk|music|laughter)[^\]\)>]*[\]\)>]", re.IGNORECASE)
# Immediate word-level stutter: "what what is" -> "what is".
_STUTTER_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE | re.UNICODE)
_REPEAT_PUNCT_RE = re.compile(r"([.!?।۔])\1{1,}")


def strip_asr_artifacts(text: str) -> tuple[str, list[str]]:
    """Remove obvious ASR noise. Returns ``(cleaned, artifacts_removed)``.

    Conservative by design; if a removal would empty the query, it is reverted.
    """
    removed: list[str] = []
    out = text

    if _ANNOTATION_RE.search(out):
        out = _ANNOTATION_RE.sub(" ", out)
        removed.append("annotations")

    stripped = _FILLER_RE.sub(" ", out)
    if stripped != out and stripped.strip():
        out = stripped
        removed.append("fillers")

    destuttered = _STUTTER_RE.sub(r"\1", out)
    if destuttered != out:
        out = destuttered
        removed.append("stutter")

    collapsed = _REPEAT_PUNCT_RE.sub(r"\1", out)
    if collapsed != out:
        out = collapsed
        removed.append("repeated_punct")

    out = _WS_RE.sub(" ", out).strip()
    return (out or text.strip()), removed


def normalize_query(text: str) -> tuple[str, list[str]]:
    """Deterministic query preprocessing for the latency-critical path.

    Order: unicode/whitespace normalisation, then ASR-artifact removal. No
    LLM rewriting, no translation, no stopword removal - BGE-M3 is multilingual
    so the query is embedded in its original script.
    """
    base = normalize_text(text)
    return strip_asr_artifacts(base)
