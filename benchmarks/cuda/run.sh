#!/usr/bin/env bash
# Build + run the TATR -> Tesseract CUDA benchmark.
#
# Designed to be run on the 4070 laptop (NVIDIA driver + nvidia-container-toolkit).
# The image can also be built on the arm64 Mac and shipped over:
#
#   # on the Mac
#   docker buildx build --platform linux/amd64 \
#     -f benchmarks/cuda/Dockerfile -t tablesalt-tatr-cuda:amd64 --load .
#   docker save tablesalt-tatr-cuda:amd64 | gzip > tablesalt-tatr-cuda-amd64.tar.gz
#
#   # on the laptop
#   gunzip -c tablesalt-tatr-cuda-amd64.tar.gz | docker load
#   ./benchmarks/cuda/run.sh --gt data/dev --out out/cuda_dev
#
# Usage:
#   ./benchmarks/cuda/run.sh [--gt data/dev] [--out out/cuda_dev] [--device cuda]
#                             [--repeat 3] [--cpu-baseline] [--build]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE="${TATR_CUDA_IMAGE:-tablesalt-tatr-cuda:amd64}"

GT="data/dev"
OUT="out/cuda_dev"
DEVICE="cuda"
REPEAT=1
CPU_BASELINE=1
BUILD=0
WORKERS="${TATR_WORKERS:-6}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt)            GT="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
    --repeat)        REPEAT="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --cpu-baseline)  CPU_BASELINE=1; shift ;;
    --no-cpu-baseline) CPU_BASELINE=0; shift ;;
    --build)         BUILD=1; shift ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$BUILD" == "1" ]]; then
  echo "== building $IMAGE (linux/amd64) =="
  docker buildx build --platform linux/amd64 \
    -f "$HERE/Dockerfile" -t "$IMAGE" --load "$ROOT"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image '$IMAGE' not found. Build it first:" >&2
  echo "  docker buildx build --platform linux/amd64 -f benchmarks/cuda/Dockerfile -t $IMAGE --load ." >&2
  exit 1
fi

mkdir -p "$ROOT/$OUT"

echo "== GPU check =="
docker run --rm --gpus all "$IMAGE" --help >/dev/null 2>&1 \
  && echo "  nvidia runtime OK" \
  || echo "  WARNING: --gpus all failed. Is the NVIDIA driver + nvidia-container-toolkit installed?"

# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------
echo
echo "== CUDA run: $GT =="
docker run --rm --gpus all \
  --network none \
  -e TATR_DEVICE="$DEVICE" \
  -v "$ROOT/data:/app/data:ro" \
  -v "$ROOT/$OUT:/app/out" \
  "$IMAGE" --gt "$GT" --out "$OUT" --device "$DEVICE" --repeat "$REPEAT" --workers "$WORKERS"

# ---------------------------------------------------------------------------
# CPU reference on the same machine (the comparison that makes the GPU number mean something)
# ---------------------------------------------------------------------------
if [[ "$CPU_BASELINE" == "1" ]]; then
  CPU_OUT="${OUT}_cpu"
  mkdir -p "$ROOT/$CPU_OUT"
  echo
  echo "== CPU reference run (same machine, same data): $CPU_OUT =="
  docker run --rm \
    --network none \
    -e TATR_DEVICE=cpu \
    -v "$ROOT/data:/app/data:ro" \
    -v "$ROOT/$CPU_OUT:/app/out" \
    "$IMAGE" --gt "$GT" --out "$CPU_OUT" --device cpu --repeat "$REPEAT" --workers "$WORKERS"
fi

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
echo
echo "== comparison =="
python3 - "$ROOT/$OUT/_bench.json" "${ROOT}/${OUT}_cpu/_bench.json" <<'PY'
import json, sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
if not paths:
    print("no _bench.json found")
    raise SystemExit(0)

runs = [json.loads(p.read_text()) for p in paths]
names = [p.parent.name for p in paths]

head = f"{'run':<24} {'device':<7} {'s/table':>9} {'p90':>8} {'TATR s':>8} {'OCR s':>8} {'FINAL':>8} {'TEDSw':>8} {'QA':>7}"
print(head)
print("-" * len(head))
for name, r in zip(names, runs):
    t = r["throughput"]
    s = r["stages"]
    sc = r.get("score", {})
    print(f"{name:<24} {r['gpu_info']['device']:<7} "
          f"{t['s_per_table_mean']:>9.3f} {t['s_per_table_p90']:>8.3f} "
          f"{s['tatr']['mean_s']:>8.4f} {s['ocr']['mean_s']:>8.3f} "
          f"{sc.get('final', 0):>8.4f} {sc.get('teds_weighted', 0):>8.4f} "
          f"{sc.get('qa_accuracy', 0):>7.3f}")

if len(runs) == 2:
    a, b = runs
    ta, tb = a["throughput"]["s_per_table_mean"], b["throughput"]["s_per_table_mean"]
    if ta and tb:
        print(f"\nspeedup (cpu/gpu)          {tb / ta:.2f}x")
    print(f"gpu info                   {a['gpu_info'].get('gpu_name', 'n/a')} "
          f"{a['gpu_info'].get('gpu_total_mem_gb', '')} GB")
    sa, sb = a.get("score", {}), b.get("score", {})
    if sa and sb:
        print(f"quality delta (gpu-cpu)    FINAL {sa.get('final', 0) - sb.get('final', 0):+.4f}  "
              f"TEDSw {sa.get('teds_weighted', 0) - sb.get('teds_weighted', 0):+.4f}")
    print(f"OCR share of wall clock     gpu {a['stages']['ocr']['share_pct']}%  "
          f"cpu {b['stages']['ocr']['share_pct']}%")
PY
