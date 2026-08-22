#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="${1:-${repo_root}/results/b200-tirx-version-suite}"
warmup="${BLACKWELL_MOE_WARMUP:-20}"
iterations="${BLACKWELL_MOE_ITERATIONS:-200}"
case_timeout="${BLACKWELL_MOE_CASE_TIMEOUT:-600}"

cd "${repo_root}"
mkdir -p "${results_dir}/generated"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

run_checked() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10s "${case_timeout}s" "$@"
  else
    "$@"
  fi
}

python3 -c 'import torch, tvm, tvm.tirx; assert torch.cuda.is_available(); print(torch.__version__, tvm.__version__, torch.cuda.get_device_name())' \
  | tee "${results_dir}/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
  --format=csv,noheader >>"${results_dir}/environment.txt"

versions=(
  v0_nonpersistent
  v0_5_persistent
  v1_static_ws
  v2_dynamic
  v3_chunked
  v4_hybrid
  v5_clc
  v6_small_m_ws
)

: >"${results_dir}/correctness.log"
for version in "${versions[@]}"; do
  run_checked python3 -m blackwell_moe_tirx.cli \
    --kernel="${version}" \
    --smoke \
    --correctness-only \
    --dump-cuda="${results_dir}/generated/${version}.cu" \
    2>&1 | tee -a "${results_dir}/correctness.log"
done
run_checked python3 -m blackwell_moe_tirx.cli \
  --kernel=v6_padding_aware \
  --smoke \
  --correctness-only \
  --dump-cuda="${results_dir}/generated/v6_padding_aware.cu" \
  2>&1 | tee -a "${results_dir}/correctness.log"

append_csv() {
  local output="$1"
  shift
  local temporary
  temporary="$(mktemp)"
  run_checked "$@" --csv >"${temporary}"
  if [[ ! -s "${output}" ]]; then
    cp "${temporary}" "${output}"
  else
    tail -n +2 "${temporary}" >>"${output}"
  fi
  rm "${temporary}"
}

results_csv="${results_dir}/versions.csv"
: >"${results_csv}"
for distribution in uniform heavy_hitter sparse zipf; do
  for version in \
    v0_nonpersistent \
    v0_5_persistent \
    v1_static_ws \
    v2_dynamic \
    v4_hybrid \
    v5_clc \
    v6_small_m_ws \
    v6_padding_aware; do
    append_csv "${results_csv}" \
      python3 -m blackwell_moe_tirx.cli \
      --kernel="${version}" \
      --distribution="${distribution}" \
      --experts=64 --tokens=4096 --n=7168 --k=2048 --seed=2026 \
      --warmup="${warmup}" --iterations="${iterations}"
  done

  for claim_size in 1 2 4 8; do
    append_csv "${results_csv}" \
      python3 -m blackwell_moe_tirx.cli \
      --kernel=v3_chunked \
      --claim-size="${claim_size}" \
      --distribution="${distribution}" \
      --experts=64 --tokens=4096 --n=7168 --k=2048 --seed=2026 \
      --warmup="${warmup}" --iterations="${iterations}"
  done
done

archive="${results_dir}.tar.gz"
tar -czf "${archive}" -C "$(dirname "${results_dir}")" \
  "$(basename "${results_dir}")"

echo "B200 TIRx version suite complete"
echo "results: ${results_dir}"
echo "archive: ${archive}"
