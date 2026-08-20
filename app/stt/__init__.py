"""Speech-to-text stage (Sarvam Saaras v3)."""

from app.stt.sarvam import (
    SarvamAuthError,
    SarvamSTT,
    SarvamSTTError,
    detect_code_mixing,
    get_stt,
    reset_stt,
    wav_header,
)

__all__ = [
    "SarvamAuthError",
    "SarvamSTT",
    "SarvamSTTError",
    "detect_code_mixing",
    "get_stt",
    "reset_stt",
    "wav_header",
]
