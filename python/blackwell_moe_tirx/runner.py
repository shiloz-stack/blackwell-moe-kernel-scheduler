"""B200 correctness and CUDA-Event benchmark harness for the TIRx kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
from typing import Any

from .config import MoEWorkloadPlan, TIRxMoESpec


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    distribution: str
    experts: int
    active_experts: int
    tokens: int
    n: int
    k: int
    tiles: int
    ctas: int
    pipe_depth: int
    useful_work_ratio: float
    median_ms: float
    p95_ms: float
    effective_tflops: float
    max_abs_error: float
    max_rel_error: float

    @staticmethod
    def csv_header() -> str:
        return (
            "device,implementation,distribution,experts,active_experts,tokens,"
            "n,k,tiles,ctas,pipe_depth,useful_work_ratio,median_ms,p95_ms,"
            "effective_tflops,max_abs_error,max_rel_error"
        )

    def csv_row(self) -> str:
        fields = (
            self.device,
            "tirx_static_persistent",
            self.distribution,
            self.experts,
            self.active_experts,
            self.tokens,
            self.n,
            self.k,
            self.tiles,
            self.ctas,
            self.pipe_depth,
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
            source = module.get_source()
        except Exception:  # Some runtime wrapper modules do not carry source.
            source = ""
        if source and ("__global__" in source or "tcgen05" in source):
            return source
        modules.extend(getattr(module, "imported_modules", []))
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


def _check_correctness(
    torch: Any,
    executable: Any,
    spec: TIRxMoESpec,
    plan: MoEWorkloadPlan,
    A: Any,
    B: Any,
    work_tiles: Any,
    D: Any,
) -> tuple[float, float]:
    D.zero_()
    executable.mod(A, B, work_tiles, D)
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
    warmup: int = 20,
    iterations: int = 200,
    correctness_only: bool = False,
    dump_cuda: Path | None = None,
) -> BenchmarkResult:
    """Compile once, enforce correctness, then report median and p95 latency."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    torch, tvm = _require_runtime()
    from .kernel import build_static_persistent_moe_kernel

    torch.backends.cuda.matmul.allow_tf32 = False
    target = tvm.target.Target("cuda -arch=sm_100a")
    kernel = build_static_persistent_moe_kernel(spec, plan)
    with target:
        executable = tvm.compile(
            tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx"
        )

    if dump_cuda is not None:
        dump_cuda.parent.mkdir(parents=True, exist_ok=True)
        dump_cuda.write_text(_module_cuda_source(executable), encoding="utf-8")

    A, B, work_tiles, D = _allocate_inputs(torch, spec, plan)
    max_abs, max_rel = _check_correctness(
        torch, executable, spec, plan, A, B, work_tiles, D
    )

    timings: list[float] = []
    if not correctness_only:
        for _ in range(warmup):
            executable.mod(A, B, work_tiles, D)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            executable.mod(A, B, work_tiles, D)
            end.record()
        torch.cuda.synchronize()
        timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]

    median_ms = statistics.median(timings) if timings else math.nan
    p95_ms = _percentile_95(timings)
    effective_tflops = (
        plan.logical_flops / (median_ms * 1.0e9) if timings else math.nan
    )
    return BenchmarkResult(
        device=torch.cuda.get_device_name(),
        distribution=plan.distribution,
        experts=spec.experts,
        active_experts=plan.active_experts,
        tokens=spec.tokens,
        n=spec.n,
        k=spec.k,
        tiles=len(plan.tiles),
        ctas=spec.cta_count,
        pipe_depth=spec.pipe_depth,
        useful_work_ratio=plan.useful_work_ratio,
        median_ms=median_ms,
        p95_ms=p95_ms,
        effective_tflops=effective_tflops,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )
