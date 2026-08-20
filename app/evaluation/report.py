"""Report writers for the evaluation and benchmark scripts.

One rule enforced here: an unmeasured value is rendered as ``n/a``, never as
``0``. A zero in a latency table reads as "impossibly fast" and is exactly how
fabricated benchmark numbers get published by accident.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["write_json", "write_csv", "write_markdown_table", "fmt", "NA"]

NA = "n/a"


def fmt(value: Any, digits: int = 2) -> str:
    """Format a cell, distinguishing 'not measured' from zero."""
    if value is None:
        return NA
    if isinstance(value, float):
        if math.isnan(value):
            return NA
        return f"{value:.{digits}f}"
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Union of keys, preserving first-seen order, so heterogeneous rows still work.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    return path


def write_markdown_table(
    rows: Sequence[dict[str, Any]], *, headers: Sequence[str] | None = None
) -> str:
    """Render rows as a GitHub-flavoured markdown table."""
    if not rows:
        return "_no data_\n"
    cols = list(headers) if headers else []
    if not cols:
        for row in rows:
            for key in row:
                if key not in cols:
                    cols.append(key)

    body = [[fmt(row.get(col)) for col in cols] for row in rows]
    try:
        from tabulate import tabulate

        return tabulate(body, headers=cols, tablefmt="github") + "\n"
    except ModuleNotFoundError:
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines) + "\n"
