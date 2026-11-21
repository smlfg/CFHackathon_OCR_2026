# CUDA benchmark: TATR → Tesseract

Container that runs the **Table Transformer (structure) → Tesseract (text)** pipeline
on an NVIDIA GPU and measures two things with numbers:

1. **Speed** — s/table, p50/p90, and a per-stage split so we know what the GPU
   actually bought us.
2. **Quality** — scored with the repo's **unmodified** `score_all.py` (the truth).

The algorithm is `scripts/my_tatr_fusion.py`, imported unchanged. This harness
only measures it — it does not re-implement anything.

---

## Quick start (on the 4070 laptop)

```bash
# prerequisites: NVIDIA driver + nvidia-container-toolkit
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi   # must print the GPU

# build (amd64; safe to run on the arm64 Mac too)
docker buildx build --platform linux/amd64 \
  -f benchmarks/cuda/Dockerfile -t tablesalt-tatr-cuda:amd64 --load .

# run + compare against a CPU run on the same machine
./benchmarks/cuda/run.sh --gt data/dev --out out/cuda_dev
```

If you build on the Mac and run on the laptop, ship the image:

```bash
docker save tablesalt-tatr-cuda:amd64 | gzip > tablesalt-tatr-cuda-amd64.tar.gz
# ...copy to laptop...
gunzip -c tablesalt-tatr-cuda-amd64.tar.gz | docker load
```

---

## What the run does

```
image  ──►  median3 denoise
             │
             ├─► TATR structure model (DETR, 28.8M) ──► rows ∩ cols ──► cells + spans
             │        ★ GPU-bound, this is what CUDA touches
             │
             └─► Tesseract per cell (deu+eng, --psm 6/7) ──► text
                      ★ CPU-bound, CUDA cannot touch this
```

Both stages are timed separately. That separation is the whole point — a fast
GPU number for stage 1 does not automatically mean a fast table.

---

## Output

| File | Content |
|---|---|
| `<out>/<table_id>.json` | cells in the official TableSalt format (`text, r, c, rs, cs`) |
| `<out>/_report.json` | per-table timings, detected vs. ground-truth shape |
| `<out>/_bench.json` | summary: throughput, stage split, GPU info, score |

`run.sh` also writes `<out>_cpu/` and prints the side-by-side comparison.

---

## Datasets

The hidden test set is not in this repo. The closest proxies that exist:

| Dataset | Tables | Role |
|---|---|---|
| `data/dev` | 24 | official yardstick, never tuned on |
| `data/val96` | 96 | generated validation set, used for decisions |
| `data/lab_promo` | 120 | generated lab set |
| `data/lab_repl_1` | 120 | generated replicate set |

```bash
./benchmarks/cuda/run.sh --gt data/val96 --out out/cuda_val96
```

---

## Reference numbers (Mac, CPU, `--workers 6`, 2026-11-21)

Measured with this harness; it reproduces the `MEASUREMENTS.md` row
"TATR → clean Tesseract" exactly (**FINAL 0.7268**, CellF1 0.6235), which is
the correctness check for this harness.

| Stage | mean s | share |
|---|---|---|
| median3 | 0.040 | 2.8 % |
| **TATR (structure)** | **0.097** | **6.8 %** |
| cells (geometry) | 0.0002 | 0.0 % |
| **Tesseract (OCR)** | **1.281** | **89.4 %** |
| **total** | **1.432** | |

s/table 1.43 (p50 1.40, p90 2.32), shape exactly right 23/24.

### What this predicts for CUDA

TATR is only **6.8 %** of the wall clock. Even an infinitely fast GPU caps the
speedup at roughly `1 / 0.068 ≈ 14.7×`, and realistically the DETR forward pass
here is ~100 ms on CPU — dominated by the ResNet-18 backbone on a 1000 px input,
not by anything the GPU is bad at. Expect:

- TATR stage: 0.097 s → ~0.010–0.020 s on the 4070
- **total s/table: ~1.35 s** (roughly 1.05–1.10×, *not* 10×)
- OCR becomes ~94 % of the wall clock

So the GPU is not the lever for this pipeline. The lever, if speed ever matters,
is the OCR stage (batched Tesseract, or a GPU OCR model). **Measure on the
laptop before believing this — it is a prediction, not a result.**

---

## Offline guarantee

The official submission run is offline. The image therefore:

- bakes the TATR checkpoint into `/app/models` at build time
  (`fetch_weights.py` — the only step that touches the network)
- ships Tesseract + `eng`/`deu` language data in the image
- sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` **after** the weights are
  baked (offline mode during the build would block the download itself)

The build ends with two preflight checks, so a broken image fails the **build**
instead of the benchmark run:

1. **Tesseract can read text.** A Tesseract without loadable language data
   returns an empty string for every crop — no error, just zeros. That is
   exactly what happened with a wrong `TESSDATA_PREFIX`: FINAL 0.16 instead of
   0.73, with a perfectly clean run log. The build now renders
   `"Total 1.234 EUR"` and asserts the digits come back.
2. **TATR loads offline.** `from_pretrained(..., local_files_only=True)` proves
   both weight sets are baked in: the detector *and* the timm ResNet-18
   backbone (timm fetches it from the Hub at construction time — easy to miss).

`run.sh` passes `--network none` to prove the run needs no network.

## Two things that silently ruin the numbers

Both were hit during development; both are now guarded.

| Trap | Symptom | Guard |
|---|---|---|
| `TESSDATA_PREFIX` points at a directory that does not exist | every cell empty, FINAL 0.16 | build-time OCR self-test + runtime check in the harness |
| Tesseract 4 vs 5 | OCR output differs → numbers not comparable with `MEASUREMENTS.md` | Ubuntu 24.04 base (Tesseract 5.3.x); the harness prints the version it used |

The harness prints the Tesseract version it ran with, so a row in
`challenge/MEASUREMENTS.md` always says which OCR produced it.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 12.6 runtime image, `linux/amd64` |
| `requirements.txt` | pinned deps, torch from the cu124 index |
| `fetch_weights.py` | bakes the TATR checkpoint into the image |
| `bench_tatr_cuda.py` | the harness: timing + official scoring |
| `run.sh` | build + GPU run + CPU reference + comparison table |
