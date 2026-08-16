"""Command-line entry point for TIRx MoE correctness and benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TIRxMoESpec, build_workload_plan
from .runner import BenchmarkResult, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and benchmark the static-persistent TIRx MoE kernel on B200"
    )
    parser.add_argument(
        "--distribution",
        choices=("uniform", "heavy_hitter", "sparse", "zipf"),
        default="uniform",
    )
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ctas", type=int, default=148)
    parser.add_argument("--pipe-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="override matrix sizes with a low-memory first-run correctness shape",
    )
    parser.add_argument("--dump-cuda", type=Path)
    parser.add_argument("--csv", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.smoke:
        args.experts = 4
        args.tokens = 128
        args.n = 256
        args.k = 64
        args.warmup = min(args.warmup, 5)
        args.iterations = min(args.iterations, 20)

    spec = TIRxMoESpec(
        experts=args.experts,
        tokens=args.tokens,
        n=args.n,
        k=args.k,
        cta_count=args.ctas,
        pipe_depth=args.pipe_depth,
    )
    plan = build_workload_plan(spec, args.distribution, seed=args.seed)
    result = run_benchmark(
        spec,
        plan,
        warmup=args.warmup,
        iterations=args.iterations,
        correctness_only=args.correctness_only,
        dump_cuda=args.dump_cuda,
    )

    if args.csv:
        print(BenchmarkResult.csv_header())
        print(result.csv_row())
    else:
        print(
            f"TIRx BF16 MoE correctness passed on {result.device}: "
            f"active_experts={result.active_experts}, tiles={result.tiles}, "
            f"max_abs_error={result.max_abs_error:.6g}, "
            f"max_rel_error={result.max_rel_error:.6g}"
        )
        if not args.correctness_only:
            print(
                f"median={result.median_ms:.6f} ms, p95={result.p95_ms:.6f} ms, "
                f"effective={result.effective_tflops:.3f} TFLOP/s, "
                f"useful_work_ratio={result.useful_work_ratio:.3%}"
            )


if __name__ == "__main__":
    main()
