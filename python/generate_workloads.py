#!/usr/bin/env python3
"""Generate deterministic MoE routing workloads without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def weights(args: argparse.Namespace) -> list[float]:
    if args.distribution == "uniform":
        return [1.0] * args.experts
    if args.distribution == "zipf":
        return [1.0 / ((rank + 1) ** args.zipf_alpha) for rank in range(args.experts)]
    if args.distribution == "heavy_hitter":
        heavy = max(1, math.ceil(args.experts * args.heavy_fraction))
        light = args.experts - heavy
        heavy_weight = args.heavy_share / heavy
        light_weight = 0.0 if light == 0 else (1.0 - args.heavy_share) / light
        return [heavy_weight if expert < heavy else light_weight for expert in range(args.experts)]
    inactive = min(args.experts - 1, math.floor(args.experts * args.inactive_fraction))
    return [0.0] * inactive + [1.0] * (args.experts - inactive)


def generate(args: argparse.Namespace) -> list[int]:
    rng = random.Random(args.seed)
    counts = [0] * args.experts
    for expert in rng.choices(range(args.experts), weights=weights(args), k=args.tokens):
        counts[expert] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", choices=("uniform", "heavy_hitter", "sparse", "zipf"), default="uniform")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--heavy-fraction", type=float, default=0.1)
    parser.add_argument("--heavy-share", type=float, default=0.8)
    parser.add_argument("--inactive-fraction", type=float, default=0.5)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.experts <= 0 or args.tokens < 0:
        parser.error("experts must be positive and tokens must be non-negative")

    counts = generate(args)
    if args.format == "json":
        rendered = json.dumps({"distribution": args.distribution, "seed": args.seed, "tokens_per_expert": counts}, indent=2) + "\n"
    else:
        rows = ["expert_id,tokens"] + [f"{expert},{tokens}" for expert, tokens in enumerate(counts)]
        rendered = "\n".join(rows) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

