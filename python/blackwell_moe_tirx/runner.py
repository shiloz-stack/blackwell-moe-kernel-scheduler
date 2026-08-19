"""B200 correctness and CUDA-Event benchmark harness for the TIRx kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Any

from .config import MoEWorkloadPlan, TIRxMoESpec
from .dispatch import routing_features


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    implementation: str
    acquisition: str
    distribution: str
    experts: int
    active_experts: int
    tokens: int
    n: int
    k: int
    tiles: int
    ctas: int
    pipe_depth: int
    claim_size: int
    hybrid_main_claim_size: int
    hybrid_tail_tiles: int
    cv_m: float
    max_over_mean_m: float
    inactive_expert_ratio: float
    small_m_expert_ratio: float
    expert_tile_cv: float
    max_expert_tiles_over_mean: float
    tiles_per_cta: float
    useful_work_ratio: float
    median_ms: float
    p95_ms: float
    effective_tflops: float
    max_abs_error: float
    max_rel_error: float

    @staticmethod
    def csv_header() -> str:
        return (
            "device,implementation,acquisition,distribution,experts,active_experts,tokens,"
            "n,k,tiles,ctas,pipe_depth,claim_size,hybrid_main_claim_size,"
            "hybrid_tail_tiles,cv_m,max_over_mean_m,inactive_expert_ratio,"
            "small_m_expert_ratio,expert_tile_cv,max_expert_tiles_over_mean,"
            "tiles_per_cta,useful_work_ratio,median_ms,p95_ms,"
            "effective_tflops,max_abs_error,max_rel_error"
        )

    def csv_row(self) -> str:
        fields = (
            self.device,
            self.implementation,
            self.acquisition,
            self.distribution,
            self.experts,
            self.active_experts,
            self.tokens,
            self.n,
            self.k,
            self.tiles,
            self.ctas,
            self.pipe_depth,
            self.claim_size,
            self.hybrid_main_claim_size,
            self.hybrid_tail_tiles,
            f"{self.cv_m:.6f}",
            f"{self.max_over_mean_m:.6f}",
            f"{self.inactive_expert_ratio:.6f}",
            f"{self.small_m_expert_ratio:.6f}",
            f"{self.expert_tile_cv:.6f}",
            f"{self.max_expert_tiles_over_mean:.6f}",
            f"{self.tiles_per_cta:.6f}",
            f"{self.useful_work_ratio:.6f}",
            f"{self.median_ms:.6f}",
            f"{self.p95_ms:.6f}",
            f"{self.effective_tflops:.3f}",
            f"{self.max_abs_error:.6g}",
            f"{self.max_rel_error:.6g}",
        )
        return ",".join(str(field) for field in fields)


def _require_runtime() -> tuple[Any, Any]:
    try:
        import torch
        import tvm
        import tvm.tirx  # noqa: F401 - verifies that this is a TIRx-enabled wheel.
    except ImportError as error:
        raise RuntimeError(
            "TIRx execution requires a CUDA PyTorch build and Apache TVM with tvm.tirx"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required")
    major, _minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("this kernel requires a Blackwell GPU (compute capability 10.x)")
    return torch, tvm


def _module_cuda_source(executable: Any) -> str:
    modules = [executable.mod]
    while modules:
        module = modules.pop()
        try:
            source = module.inspect_source()
        except Exception:
            try:
                source = module.get_source()
            except Exception:  # Runtime wrapper modules may not carry source.
                source = ""
        if source and ("__global__" in source or "tcgen05" in source):
            return source
        try:
            imports = module.imports_
        except (AttributeError, TypeError):
            imports = getattr(module, "imported_modules", ())
        modules.extend(imports)
    raise RuntimeError("compiled module did not expose generated CUDA source")


def _percentile_95(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _allocate_inputs(torch: Any, spec: TIRxMoESpec, plan: MoEWorkloadPlan):
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(plan.seed)

    # Zero padding makes every 128-row TMA load legal without changing the
    # mathematical result for an expert's valid routed-token rows.
    A = torch.zeros(
        (spec.experts, plan.m_capacity, spec.k),
        dtype=torch.bfloat16,
        device=device,
    )
    B = torch.randn(
        (spec.experts, spec.n, spec.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) * 0.25
    for expert_id, expert_m in enumerate(plan.tokens_per_expert):
        if expert_m:
            A[expert_id, :expert_m, :] = (
                torch.randn(
                    (expert_m, spec.k),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * 0.25
            )
    D = torch.zeros(
        (spec.experts, plan.m_capacity, spec.n),
        dtype=torch.bfloat16,
        device=device,
    )
    work_tiles = torch.tensor(plan.device_worklist(), dtype=torch.int32, device=device)
    return A, B, work_tiles, D


def _allocate_queue(torch: Any, descriptor: Any, spec: TIRxMoESpec, plan: MoEWorkloadPlan):
    if not descriptor.requires_queue:
        return None, None
    initial_values = descriptor.queue_initial_values(spec, plan)
    initial = torch.tensor(initial_values, dtype=torch.int32, device="cuda")
    queue_heads = initial.clone()
    return queue_heads, initial


def _invoke(
    executable: Any,
    descriptor: Any,
    A: Any,
    B: Any,
    work_tiles: Any,
    queue_heads: Any,
    D: Any,
) -> None:
    if descriptor.requires_queue:
        executable(A, B, work_tiles, queue_heads, D)
    else:
        executable(A, B, work_tiles, D)


def _check_correctness(
    torch: Any,
    executable: Any,
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    A: Any,
    B: Any,
    work_tiles: Any,
    queue_heads: Any,
    queue_initial: Any,
    descriptor: Any,
    D: Any,
) -> tuple[float, float]:
    D.zero_()
    if queue_heads is not None:
        queue_heads.copy_(queue_initial)
    _invoke(executable, descriptor, A, B, work_tiles, queue_heads, D)
    torch.cuda.synchronize()

    max_abs = 0.0
    max_rel = 0.0
    for expert_id, expert_m in enumerate(plan.tokens_per_expert):
        if not expert_m:
            continue
        reference = (A[expert_id, :expert_m, :].float() @ B[expert_id].float().T).to(
            torch.bfloat16
        )
        actual = D[expert_id, :expert_m, :]
        difference = (actual.float() - reference.float()).abs()
        denominator = reference.float().abs().clamp_min(1.0e-6)
        max_abs = max(max_abs, float(difference.max().item()))
        max_rel = max(max_rel, float((difference / denominator).max().item()))
        torch.testing.assert_close(actual, reference, rtol=2.0e-2, atol=5.0e-2)
    return max_abs, max_rel


def run_benchmark(
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    *,
    version: str = "v1_static_ws",
    warmup: int = 20,
    iterations: int = 200,
    correctness_only: bool = False,
    dump_cuda: Path | None = None,
) -> BenchmarkResult:
    """Compile once, enforce correctness, then report median and p95 latency."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    torch, tvm = _require_runtime()
    from .kernels.registry import get_descriptor

    torch.backends.cuda.matmul.allow_tf32 = False
    descriptor = get_descriptor(version)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    kernel = descriptor.build(spec, plan)
    with target:
        executable = tvm.compile(
            tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx"
        )

    if dump_cuda is not None:
        dump_cuda.parent.mkdir(parents=True, exist_ok=True)
        dump_cuda.write_text(_module_cuda_source(executable), encoding="utf-8")

    A, B, work_tiles, D = _allocate_inputs(torch, spec, plan)
    queue_heads, queue_initial = _allocate_queue(torch, descriptor, spec, plan)
    max_abs, max_rel = _check_correctness(
        torch,
        executable,
        spec,
        plan,
        A,
        B,
        work_tiles,
        queue_heads,
        queue_initial,
        descriptor,
        D,
    )

    timings: list[float] = []
    if not correctness_only:
        for _ in range(warmup):
            if queue_heads is not None:
                queue_heads.copy_(queue_initial)
            _invoke(executable, descriptor, A, B, work_tiles, queue_heads, D)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            # Queue initialization intentionally precedes the start event.  CSV
            # latency isolates in-kernel scheduling; end-to-end dispatch can
            # report queue reset separately if it becomes material.
            if queue_heads is not None:
                queue_heads.copy_(queue_initial)
            start.record()
            _invoke(executable, descriptor, A, B, work_tiles, queue_heads, D)
            end.record()
        torch.cuda.synchronize()
        timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]

    median_ms = statistics.median(timings) if timings else math.nan
    p95_ms = _percentile_95(timings)
    effective_tflops = (
        plan.logical_flops / (median_ms * 1.0e9) if timings else math.nan
    )
    features = routing_features(spec, plan)
    return BenchmarkResult(
        device=torch.cuda.get_device_name(),
        implementation=descriptor.implementation,
        acquisition=descriptor.acquisition.replace(",", ";"),
        distribution=plan.distribution,
        experts=spec.experts,
        active_experts=plan.active_experts,
        tokens=spec.tokens,
        n=spec.n,
        k=spec.k,
        tiles=len(plan.tiles),
        ctas=descriptor.launch_ctas(spec, plan),
        pipe_depth=spec.pipe_depth,
        claim_size=spec.claim_size,
        hybrid_main_claim_size=spec.hybrid_main_claim_size,
        hybrid_tail_tiles=spec.hybrid_tail_tiles,
        cv_m=features.cv_m,
        max_over_mean_m=features.max_over_mean_m,
        inactive_expert_ratio=features.inactive_expert_ratio,
        small_m_expert_ratio=features.small_m_expert_ratio,
        expert_tile_cv=features.expert_tile_cv,
        max_expert_tiles_over_mean=features.max_expert_tiles_over_mean,
        tiles_per_cta=features.tiles_per_cta,
        useful_work_ratio=plan.useful_work_ratio,
        median_ms=median_ms,
        p95_ms=p95_ms,
        effective_tflops=effective_tflops,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )
