"""Language code handling for MSMARCO-XI and Sarvam.

Three distinct code spaces have to be reconciled, and conflating them is a
common source of silent bugs:

1. **ISO-639-1** (``hi``)          - what the application/API speaks.
2. **MSMARCO-XI file codes**       - the dataset's own 3-letter filename stems
   (``hin`` in ``train/hintrain.parquet``). These are *not* consistently
   ISO-639-3 and are *not* derivable by truncation (Odia is ``ori``, Punjabi
   ``pan``, Telugu ``tel``).
3. **Sarvam BCP-47-ish tags** (``hi-IN``) - what the STT service returns.

Verified against the live repo listing of ``ai4bharat/MSMARCO-XI`` (2026-08):
``train/`` contains asm ben guj hin kan mal mar nep ori pan san tam urd and
``validation/`` contains those plus ``tel``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LanguageSpec",
    "LANGUAGES",
    "SUPPORTED_ISO1",
    "TRAIN_LANGUAGES",
    "VALIDATION_LANGUAGES",
    "dataset_filename",
    "normalize_language",
    "sarvam_to_iso1",
    "iso1_to_sarvam",
    "script_of",
    "language_name",
    "has_split",
]


@dataclass(frozen=True)
class LanguageSpec:
    iso1: str
    """ISO-639-1 two-letter code, the application's canonical form."""

    file_code: str
    """3-letter stem used in MSMARCO-XI parquet filenames."""

    name: str
    sarvam_tag: str
    script: str
    has_train: bool
    has_validation: bool


# NOTE `has_train=False` for Telugu is a real upstream gap, not a typo. The
# dataset README advertises `teltrain.jsonl` but no such parquet exists in the
# repo, so Telugu can only be ingested from the validation split.
LANGUAGES: dict[str, LanguageSpec] = {
    "as": LanguageSpec("as", "asm", "Assamese", "as-IN", "Beng", True, True),
    "bn": LanguageSpec("bn", "ben", "Bengali", "bn-IN", "Beng", True, True),
    "gu": LanguageSpec("gu", "guj", "Gujarati", "gu-IN", "Gujr", True, True),
    "hi": LanguageSpec("hi", "hin", "Hindi", "hi-IN", "Deva", True, True),
    "kn": LanguageSpec("kn", "kan", "Kannada", "kn-IN", "Knda", True, True),
    "ml": LanguageSpec("ml", "mal", "Malayalam", "ml-IN", "Mlym", True, True),
    "mr": LanguageSpec("mr", "mar", "Marathi", "mr-IN", "Deva", True, True),
    "ne": LanguageSpec("ne", "nep", "Nepali", "ne-NP", "Deva", True, True),
    "or": LanguageSpec("or", "ori", "Odia", "od-IN", "Orya", True, True),
    "pa": LanguageSpec("pa", "pan", "Punjabi", "pa-IN", "Guru", True, True),
    "sa": LanguageSpec("sa", "san", "Sanskrit", "sa-IN", "Deva", True, True),
    "ta": LanguageSpec("ta", "tam", "Tamil", "ta-IN", "Taml", True, True),
    "te": LanguageSpec("te", "tel", "Telugu", "te-IN", "Telu", False, True),
    "ur": LanguageSpec("ur", "urd", "Urdu", "ur-IN", "Arab", True, True),
    # English is not a MSMARCO-XI file; it is an *optional* derived
    # representation built from the `English_passages` column for cross-lingual
    # fallback. It therefore has no split of its own.
    "en": LanguageSpec("en", "eng", "English", "en-IN", "Latn", False, False),
}

SUPPORTED_ISO1: tuple[str, ...] = tuple(LANGUAGES)
TRAIN_LANGUAGES: tuple[str, ...] = tuple(k for k, v in LANGUAGES.items() if v.has_train)
VALIDATION_LANGUAGES: tuple[str, ...] = tuple(
    k for k, v in LANGUAGES.items() if v.has_validation
)

_SARVAM_TO_ISO1: dict[str, str] = {v.sarvam_tag.lower(): k for k, v in LANGUAGES.items()}
# Sarvam has used a couple of non-obvious tags for Odia over time; accept both.
_SARVAM_TO_ISO1.update({"or-in": "or", "od-in": "or", "en-us": "en", "en-gb": "en"})

_SPLIT_SUFFIX = {"train": "train", "validation": "val"}

# Unicode block ranges good enough for coarse script identification of the
# Indic scripts in this dataset. Used only as a cheap sanity signal, never as a
# replacement for Sarvam's language detection.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0600, 0x06FF, "Arab"),
    (0x0041, 0x007A, "Latn"),
)


def normalize_language(code: str | None) -> str | None:
    """Coerce any of the three code spaces into a canonical ISO-639-1 code.

    Returns ``None`` for unknown/empty input so callers can decide whether to
    fall back to cross-lingual retrieval instead of guessing a language.
    """
    if not code:
        return None
    raw = code.strip().lower().replace("_", "-")
    if not raw:
        return None
    if raw in LANGUAGES:
        return raw
    if raw in _SARVAM_TO_ISO1:
        return _SARVAM_TO_ISO1[raw]
    # `hin`, `hin_Deva`, `hi-IN`, `hi-in-x-foo` all resolve here.
    head = raw.split("-")[0]
    if head in LANGUAGES:
        return head
    for spec in LANGUAGES.values():
        if head == spec.file_code:
            return spec.iso1
    return None


def sarvam_to_iso1(tag: str | None) -> str | None:
    """Map a Sarvam language tag (``hi-IN``) to ISO-639-1 (``hi``)."""
    return normalize_language(tag)


def iso1_to_sarvam(iso1: str) -> str:
    spec = LANGUAGES.get(iso1)
    return spec.sarvam_tag if spec else "unknown"


def language_name(iso1: str | None) -> str:
    if not iso1:
        return "unknown"
    spec = LANGUAGES.get(iso1)
    return spec.name if spec else iso1


def has_split(iso1: str, split: str) -> bool:
    """Whether this language actually exists in the given upstream split."""
    spec = LANGUAGES.get(iso1)
    if spec is None:
        return False
    if split == "train":
        return spec.has_train
    if split == "validation":
        return spec.has_validation
    return False


def dataset_filename(iso1: str, split: str) -> str:
    """Build the MSMARCO-XI repo-relative parquet path for a language/split.

    ``("hi", "train") -> "train/hintrain.parquet"``
    """
    spec = LANGUAGES.get(iso1)
    if spec is None:
        raise KeyError(f"Unsupported language {iso1!r}; expected one of {SUPPORTED_ISO1}")
    if split not in _SPLIT_SUFFIX:
        raise ValueError(f"Unsupported split {split!r}; expected 'train' or 'validation'")
    if not has_split(iso1, split):
        raise FileNotFoundError(
            f"MSMARCO-XI has no {split!r} file for {spec.name} ({iso1}). "
            f"Available: train={spec.has_train}, validation={spec.has_validation}."
        )
    return f"{split}/{spec.file_code}{_SPLIT_SUFFIX[split]}.parquet"


def script_of(text: str, sample_chars: int = 400) -> str | None:
    """Dominant Unicode script of ``text``, ignoring punctuation/digits."""
    counts: dict[str, int] = {}
    for ch in text[:sample_chars]:
        cp = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
