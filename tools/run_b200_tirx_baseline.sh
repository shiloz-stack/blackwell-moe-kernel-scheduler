#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p results
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c 'import torch, tvm, tvm.tirx; assert torch.cuda.is_available(); print(torch.__version__, tvm.__version__, torch.cuda.get_device_name())'

python3 -m blackwell_moe_tirx.cli \
  --smoke \
  --correctness-only \
  --dump-cuda results/tirx_static_persistent_sm100a.cu \
  2>&1 | tee results/b200_tirx_smoke.log

for distribution in uniform heavy_hitter sparse zipf; do
  python3 -m blackwell_moe_tirx.cli \
    --distribution="${distribution}" \
    --experts=64 \
    --tokens=4096 \
    --n=7168 \
    --k=2048 \
    --seed=2026 \
    --warmup=20 \
    --iterations=200 \
    --csv \
    2>&1 | tee "results/b200_tirx_${distribution}.csv"
done

tar -czf results/b200-tirx-static-persistent-results.tar.gz \
  results/b200_tirx_smoke.log \
  results/b200_tirx_uniform.csv \
  results/b200_tirx_heavy_hitter.csv \
  results/b200_tirx_sparse.csv \
  results/b200_tirx_zipf.csv \
  results/tirx_static_persistent_sm100a.cu

echo "Wrote results/b200-tirx-static-persistent-results.tar.gz"
