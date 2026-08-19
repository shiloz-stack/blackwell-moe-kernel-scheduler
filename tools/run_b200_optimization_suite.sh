#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-gpu-make}"
results_dir="${2:-results/b200-optimization-suite}"
warmup="${BLACKWELL_MOE_WARMUP:-20}"
iterations="${BLACKWELL_MOE_ITERATIONS:-200}"

mkdir -p "${results_dir}"

append_csv() {
  local output="$1"
  shift
  local temporary
  temporary="$(mktemp)"
  "$@" --csv >"${temporary}"
  if [[ ! -s "${output}" ]]; then
    cp "${temporary}" "${output}"
  else
    tail -n +2 "${temporary}" >>"${output}"
  fi
  rm "${temporary}"
}

for binary in \
  test_cutlass_baseline_correctness \
  test_scheduler_probe \
  test_sm100_dense_correctness \
  test_sm100_2sm_correctness \
  test_direct_cute_correctness \
  moe_cutlass_baseline_bench \
  moe_scheduler_gpu_bench \
  sm100_dense_bench \
  sm100_2sm_bench \
  direct_cute_bench; do
  if [[ ! -x "${build_dir}/${binary}" ]]; then
    echo "missing executable: ${build_dir}/${binary}" >&2
    echo "configure with BLACKWELL_MOE_ENABLE_CUDA=ON and " \
         "BLACKWELL_MOE_ENABLE_SM100_NATIVE=ON, then rebuild" >&2
    exit 1
  fi
done

{
  date -u
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
    --format=csv,noheader
  nvcc --version
  cmake --version
} >"${results_dir}/environment.txt"

ctest --test-dir "${build_dir}" --output-on-failure \
  2>&1 | tee "${results_dir}/correctness.log"

baseline_csv="${results_dir}/grouped_baseline.csv"
: >"${baseline_csv}"
for distribution in uniform heavy_hitter sparse zipf; do
  append_csv "${baseline_csv}" "${build_dir}/moe_cutlass_baseline_bench" \
    "--distribution=${distribution}" \
    --experts=64 --tokens=4096 --n=7168 --k=2048 \
    "--warmup=${warmup}" "--iterations=${iterations}"
done

native_csv="${results_dir}/native_1sm_vs_2sm.csv"
: >"${native_csv}"
for m in 128 256 512 1024 4096 8192; do
  append_csv "${native_csv}" "${build_dir}/sm100_dense_bench" \
    "--m=${m}" --n=7168 --k=2048 \
    "--warmup=${warmup}" "--iterations=${iterations}"
  append_csv "${native_csv}" "${build_dir}/sm100_2sm_bench" \
    "--m=${m}" --n=7168 --k=2048 \
    "--warmup=${warmup}" "--iterations=${iterations}"
done

direct_csv="${results_dir}/direct_cute.csv"
: >"${direct_csv}"
for m in 128 256 512 1024; do
  append_csv "${direct_csv}" "${build_dir}/direct_cute_bench" \
    "--m=${m}" --n=7168 --k=2048 \
    "--warmup=${warmup}" "--iterations=${iterations}"
done

scheduler_csv="${results_dir}/scheduler_probe.csv"
: >"${scheduler_csv}"
for distribution in uniform heavy_hitter sparse zipf; do
  append_csv "${scheduler_csv}" "${build_dir}/moe_scheduler_gpu_bench" \
    "--distribution=${distribution}" --scheduler=static \
    --experts=64 --tokens=4096 --n=7168 --tile-m=128 --tile-n=128 \
    --ctas=120 --claim-size=1 --work-scale=8 \
    "--warmup=${warmup}" "--iterations=${iterations}"
  for claim_size in 1 2 4 8; do
    append_csv "${scheduler_csv}" "${build_dir}/moe_scheduler_gpu_bench" \
      "--distribution=${distribution}" --scheduler=dynamic \
      --experts=64 --tokens=4096 --n=7168 --tile-m=128 --tile-n=128 \
      --ctas=120 "--claim-size=${claim_size}" --work-scale=8 \
      "--warmup=${warmup}" "--iterations=${iterations}"
  done
done

tar -czf "${results_dir}.tar.gz" -C "$(dirname "${results_dir}")" \
  "$(basename "${results_dir}")"

echo "B200 optimization suite complete"
echo "results: ${results_dir}"
echo "archive: ${results_dir}.tar.gz"
