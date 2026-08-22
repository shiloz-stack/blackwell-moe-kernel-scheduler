"""Command-line entry point for TIRx MoE correctness and benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    TIRxMoESpec,
    build_padding_aware_plans,
    build_workload_plan,
)
from .dispatch import routing_features, select_kernel
from .kernels import list_versions
from .runner import (
    BenchmarkResult,
    run_benchmark,
    run_padding_aware_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and benchmark a versioned TIRx MoE kernel on B200"
    )
    parser.add_argument(
        "--kernel",
        choices=("auto", "v6_padding_aware", *list_versions()),
        default="v1_static_ws",
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
    parser.add_argument("--claim-size", type=int, default=4)
    parser.add_argument("--hybrid-main-claim-size", type=int, default=8)
    parser.add_argument(
        "--hybrid-tail-tiles",
        type=int,
        default=296,
        help="V4 tiles reserved for claim-1 tail acquisition",
    )
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
        # Four K tiles exercise both stages and a complete phase wrap.
        args.k = 256
        args.warmup = min(args.warmup, 5)
        args.iterations = min(args.iterations, 20)

    tile_m = 64 if args.kernel == "v6_small_m_ws" else 128
    spec = TIRxMoESpec(
        experts=args.experts,
        tokens=args.tokens,
        n=args.n,
        k=args.k,
        cta_count=args.ctas,
        pipe_depth=args.pipe_depth,
        claim_size=args.claim_size,
        hybrid_main_claim_size=args.hybrid_main_claim_size,
        hybrid_tail_tiles=args.hybrid_tail_tiles,
        tile_m=tile_m,
    )
    if args.kernel == "v6_padding_aware":
        bucketed = build_padding_aware_plans(
            spec, args.distribution, seed=args.seed
        )
        plan = build_workload_plan(
            bucketed.large_spec,
            args.distribution,
            seed=args.seed,
            token_counts=bucketed.large.tokens_per_expert,
        )
        selected_kernel = args.kernel
        result = run_padding_aware_benchmark(
            bucketed,
            warmup=args.warmup,
            iterations=args.iterations,
            correctness_only=args.correctness_only,
            dump_cuda=args.dump_cuda,
        )
    else:
        plan = build_workload_plan(spec, args.distribution, seed=args.seed)
        selected_kernel = (
            select_kernel(spec, plan) if args.kernel == "auto" else args.kernel
        )
        result = run_benchmark(
            spec,
            plan,
            version=selected_kernel,
            warmup=args.warmup,
            iterations=args.iterations,
            correctness_only=args.correctness_only,
            dump_cuda=args.dump_cuda,
        )

    if args.csv:
        print(BenchmarkResult.csv_header())
        print(result.csv_row())
    else:
        features = routing_features(spec, plan)
        print(
            f"TIRx BF16 MoE correctness passed on {result.device}: "
            f"kernel={selected_kernel}, "
            f"active_experts={result.active_experts}, tiles={result.tiles}, "
            f"tile_m_path={result.tile_m_path}, "
            f"max_abs_error={result.max_abs_error:.6g}, "
            f"max_rel_error={result.max_rel_error:.6g}"
        )
        if not args.correctness_only:
            print(
                f"median={result.median_ms:.6f} ms, p95={result.p95_ms:.6f} ms, "
                f"effective={result.effective_tflops:.3f} TFLOP/s, "
                f"useful_work_ratio={result.useful_work_ratio:.3%}, "
                f"padding_reduction={result.padding_reduction:.3%}"
            )
        if args.kernel == "auto":
            print(
                "bootstrap dispatch features: "
                f"cv_m={features.cv_m:.3f}, "
                f"inactive_experts={features.inactive_expert_ratio:.3%}, "
                f"tiles_per_cta={features.tiles_per_cta:.3f}"
            )


if __name__ == "__main__":
    main()
