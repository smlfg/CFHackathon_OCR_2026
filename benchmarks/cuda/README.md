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

### Option A: build on the laptop (recommended)

Native amd64, no emulation, fastest build. Needs network for the pip wheels
and the two model weight sets — ~4 GB, of which ~2.5 GB is the torch CUDA wheel.

### Option B: build on the Mac, ship the image

The image is already built and verified here, so this saves the laptop the
download entirely:

```bash
# on the Mac
docker save tablesalt-tatr-cuda:amd64 | gzip > tablesalt-tatr-cuda-amd64.tar.gz
scp tablesalt-tatr-cuda-amd64.tar.gz <laptop>:

# on the laptop
gunzip -c tablesalt-tatr-cuda-amd64.tar.gz | docker load
docker images | grep tablesalt-tatr-cuda     # must be listed

./benchmarks/cuda/run.sh --gt data/dev --out out/cuda_dev
```

The image is **5.7 GB** and barely compresses (the torch CUDA wheels are already
compressed), so `gzip` saves almost nothing — a plain `docker save` over a USB
stick or LAN is the practical way to move it. It is `linux/amd64`, which is what
the laptop needs.

Option A (build on the laptop) avoids the transfer entirely and is ~4 GB of
downloads, of which ~2.5 GB is the torch CUDA wheel.

### Option C: from the repo, no image transfer

If the laptop has the repo checked out and network access, `run.sh --build`
does everything:

```bash
git pull
./benchmarks/cuda/run.sh --build --gt data/dev --out out/cuda_dev
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
| `<out>/_bench.json` | summary: throughput, stage split, GPU info, Tesseract version, score |

`run.sh` also writes `<out>_cpu/` and prints the side-by-side comparison.

### Mount layout

`run.sh` mounts the repo at **`/repo`**, not at `/app`. `/app` belongs to the
image (harness + baked weights), and mounting over it would hide both. The
paths inside the container are therefore the same relative paths as on the
host, and the results land in `<out>/` — not nested twice.

```
docker run --rm --gpus all --network none \
  -v "$PWD:/repo" -w /repo \
  tablesalt-tatr-cuda:amd64 --gt data/dev --out out/cuda_dev
```

The scorer runs from the mounted repo, so the numbers come from the repo's
`score_all.py` / `evaluate.py` / `eval_qa.py` — not from an older copy in the
image.

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

### Verified in the container (CPU path, `tablesalt-tatr-cuda:amd64`)

Full dev run inside the image, offline (`--network none`), Ubuntu 24.04 +
Tesseract **5.3.4**:

| | T0 | T1 | T2 | T3 | QA | **FINAL** |
|---|---|---|---|---|---|---|
| container (Tess 5.3.4) | 0.898 | 0.868 | 0.752 | 0.648 | 0.508 | **0.7124** |
| Mac host (Tess 5.5.3) | 0.899 | 0.867 | 0.764 | 0.682 | 0.508 | **0.7268** |

The gap is the Tesseract version (T3: 0.648 vs 0.682), not the container. The
image runs the pipeline correctly; s/table is ~3.8 there only because amd64 is
emulated on an arm64 Mac.

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

### A note on the self-test string

The self-test reads `Total 1234 EUR` and not `Total 1.234 EUR`. The German
Tesseract model reads `1.234` as `1234` — it treats the dot as a thousands
separator. That is the known `number-format` error class in
[`challenge/ERRORS.md`](../../challenge/ERRORS.md), not a broken OCR setup, so
the self-test deliberately avoids a decimal point. It only has to prove that
Tesseract reads *something* correctly.


---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 12.6 runtime image, `linux/amd64` |
| `requirements.txt` | pinned deps, torch from the cu124 index |
| `fetch_weights.py` | bakes the TATR checkpoint into the image |
| `bench_tatr_cuda.py` | the harness: timing + official scoring |
| `run.sh` | build + GPU run + CPU reference + comparison table |
