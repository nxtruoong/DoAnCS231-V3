# Kaggle notebook — Run 9

One notebook: `run9.ipynb`

End-to-end recipe on Kaggle T4×2 for **ResNet-50 (ImageNet pretrained) + CBAM + MediaPipe pose fusion**.

## Quick start

1. Create a Kaggle notebook, attach the State Farm competition dataset, set GPU = T4×2, Internet = On.
2. Make scripts importable — either:
   - **Option A (recommended):** upload the repo as a Kaggle dataset at `/kaggle/input/driver-distraction-cbam`, prepend to `sys.path` in cell 1.
   - **Option B:** `!git clone <repo-url> /kaggle/working/code` and use that as `CODE_DIR`.
3. Open `run9.ipynb` and run cells top-to-bottom.

## What the notebook does

| Cell | Purpose | Time |
|------|---------|------|
| 1a | Paths + sanity asserts | — |
| 1b | `data_prep.py` — subject-wise splits + dataset stats | ~10 min |
| 1c | Install mediapipe + polars | ~2 min |
| 2 | `extract_pose.py` — precompute 36-d pose vectors | ~18 min |
| 3 | Smoke test — 2 ep ResNet-50 pretrained | ~5–7 min |
| 4 | Full Run 9 training — 40 ep, early stop | ~2–2.5 hr |
| 5 | Resume if interrupted | — |
| 6 | Peek at history | — |
| 7 | `eval_twostream.py` — metrics + figures | ~5 min |
| 8–9 | Compare to prior runs, per-class breakdown | — |
| 10 | Bundle artifacts for download | — |

## Expected result

Accuracy **~0.949**, macro F1 **~0.948** on subject-wise val (5 held-out drivers).

## Prior runs

See `RUNS_ARCHIVE.md` for results from Runs 1–8 and 10.
