#!/usr/bin/env python
"""Pre-fetch the three local models so the first request is not a 5GB download.

Skips the ONNX variants in each repo (another ~3.6GB that this implementation
does not use), which roughly halves the download.

    python scripts/download_models.py
    python scripts/download_models.py --only embedding
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

# Weight files + tokenizer assets only. `onnx/**` and `imgs/**` are excluded.
COMMON = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "spm.model",
    "added_tokens.json",
]

TARGETS = {
    # bge-m3 publishes no safetensors; pytorch_model.bin is the only weight file.
    # sparse_linear.pt is the learned sparse head - without it there is no
    # lexical/sparse representation and "hybrid" retrieval would be dense-only.
    "embedding": (["pytorch_model.bin", "sparse_linear.pt", *COMMON]),
    "reranker": (["model.safetensors", *COMMON]),
    "nli": (["model.safetensors", *COMMON]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(TARGETS), help="Fetch a single model.")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    settings = get_settings()
    repos = {
        "embedding": (settings.embedding_model, settings.embedding_model_revision),
        "reranker": (settings.reranker_model, settings.reranker_model_revision),
        "nli": (settings.nli_model, settings.nli_model_revision),
    }
    if args.only:
        repos = {args.only: repos[args.only]}

    failures = 0
    for role, (repo_id, revision) in repos.items():
        print(f"\n=== {role}: {repo_id}@{revision[:12]} ===", flush=True)
        started = time.time()
        try:
            # Pinned revision. Without it, this prefetch and the later
            # from_pretrained() can resolve `main` to different commits and
            # download the weights twice (2.3GB each).
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=TARGETS[role],
                token=settings.hf_token or None,
                max_workers=4,
            )
            print(f"  ok in {time.time() - started:.0f}s -> {path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
