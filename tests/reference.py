#!/usr/bin/env python3
"""Small dependency-free reference checks for generated routing workloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = payload["tokens_per_expert"]
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts):
        raise SystemExit("invalid tokens_per_expert")
    print(f"experts={len(counts)} total_tokens={sum(counts)} active={sum(value > 0 for value in counts)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} workload.json")
    main(sys.argv[1])

