# RUN9_HOWTO — ResNet-50 (ImageNet) + CBAM + MediaPipe pose fusion

End-to-end Kaggle T4×2 recipe for Run 9: the project's best model.

**Architecture:** ResNet-50 (ImageNet1K_V2 pretrained) + CBAM after each of 4 stages
(2048-d features) fused with a 36-d MediaPipe Pose vector via MLP(36→128→128),
concat 2176 → FC(2176→256→10).

**Result:** accuracy **0.949**, macro F1 **0.948**, weighted F1 **0.949** on
subject-wise val (4524 images, 5 held-out drivers).

See `RUNS_ARCHIVE.md` for comparison with earlier runs.

---

## 0. Prerequisites

- Kaggle account phone-verified (required for GPU). Competition rules accepted once:
  `https://www.kaggle.com/c/state-farm-distracted-driver-detection`
- Code available on Kaggle as either:
  - **Option A (recommended):** repo uploaded as a Kaggle dataset at
    `/kaggle/input/driver-distraction-cbam`
  - **Option B:** `!git clone https://github.com/nxtruoong/DoAnCS231-V2 /kaggle/working/code`

Required files in `CODE_DIR`:
- `data_prep.py`, `extract_pose.py`
- `model.py`, `model_resnet50.py`, `model_twostream.py`
- `augment.py`, `augment_twostream.py`
- `train_twostream.py`, `eval_twostream.py`, `eval.py`

---

## 1a. Paths

```python
import os, sys
COMP_DIR = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
WORK     = "/kaggle/working"
RUN      = f"{WORK}/run9"

assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(CODE_DIR + "/model_resnet50.py"),     "Run 9 code missing"
sys.path.insert(0, CODE_DIR)
print("OK. GPU count:", __import__("torch").cuda.device_count())
```

---

## 1b. Data prep — only if `splits/stats.json` missing

```python
import subprocess, os
if not os.path.exists(f"{WORK}/splits/stats.json"):
    subprocess.run([
        "python", f"{CODE_DIR}/data_prep.py",
        "--data-root", COMP_DIR,
        "--out-dir",   f"{WORK}/splits",
        "--batch-size", "64",
        "--num-workers", "4",
    ], check=True)
else:
    print("splits already exist, skipping")
```

---

## 1c. Install MediaPipe + polars — only if `pose.parquet` missing

Pin MediaPipe to 0.10.13. Newer versions lazy-load `mp.solutions` and break on
Kaggle Python 3.12 when TF/XLA registers CUDA factories first.

```python
import os
if not os.path.exists(f"{WORK}/splits/pose.parquet"):
    !pip install --no-cache-dir mediapipe==0.10.13 polars
    !python -c "from mediapipe.python.solutions import pose as mp_pose; print('OK,', mp_pose.Pose)"
else:
    print("pose.parquet already exists; skipping mediapipe install")
    !pip install --no-cache-dir polars
```

---

## 2. Precompute pose features — one-time, ~18 min

Runs MediaPipe Pose over every training image. Writes `splits/pose.parquet`:
36 features per image (head, wrists, elbows, fingers, hips, derived signals,
visibility gates). See `extract_pose.py` docstring for the full feature index.

```python
import subprocess, os
if not os.path.exists(f"{WORK}/splits/pose.parquet"):
    subprocess.run([
        "python", f"{CODE_DIR}/extract_pose.py",
        "--img-root", f"{COMP_DIR}/imgs/train",
        "--out",      f"{WORK}/splits/pose.parquet",
    ], check=True)
else:
    print("pose.parquet already exists, skipping")
```

**Pass criterion:** detection rate ≥ 0.70 (printed at end of script).

---

## 3. Smoke test — 2 epochs, ~5–7 min

Verifies ImageNet pretrained load + imagenet stats + pose fusion wire correctly
before committing to the full 2.5 hr run.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
    "--backbone", "resnet50",
    "--pretrained",
    "--imagenet-stats",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run9_smoke",
    "--epochs", "2",
    "--batch-size", "32",
    "--num-workers", "2",
    "--lr", "0.01",
    "--full-size", "320",
    "--data-parallel",
], check=True)
```

**Expect:** 2 epochs complete, no OOM. Pretrained ResNet-50 starts hot — val acc
should already be 0.40–0.70 after 2 ep (vs Run 8 ep 2 ~0.20).

**OOM:** drop `--batch-size 24` or `--full-size 288`.

---

## 3b. Restart kernel before full training (recommended)

Smoke test leaves MediaPipe + worker processes in memory. Restart the kernel
(Run → Restart & Clear Cell Outputs), re-run Cell 1a only. Skip 1b / 1c / 2 / 3
(outputs already on disk), go straight to Cell 4.

---

## 4. Full training — ~2–2.5 hr

LR note: pretrained backbone fine-tunes at 0.01 (vs Run 8's 0.03 from scratch).

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
    "--backbone", "resnet50",
    "--pretrained",
    "--imagenet-stats",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "40",
    "--batch-size", "32",
    "--num-workers", "2",
    "--lr", "0.01",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--full-size", "320",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

**Use Save Version → Save & Run All — Commit** so the run survives browser disconnects.

**Milestones (pretrained beats Run 8 early):**

| ep | target ema val acc | Run 8 actual |
|---:|---:|---:|
| 2  | ≥ 0.55 | ~0.20 |
| 10 | ≥ 0.85 | ~0.78 |
| 20 | ≥ 0.89 | ~0.85 |
| 30 | ≥ 0.91 | ~0.87 |
| end | ≥ 0.93 | ~0.88 |

**OOM fixes:**
- Batch 32 / 320 px OOM → `--batch-size 24` or `--full-size 288`
- Host RAM > 10 GB by ep 3 → `--num-workers 0` or drop `--data-parallel`

---

## 5. Resume if interrupted

```python
import subprocess, glob
ckpts = sorted(glob.glob(f"{RUN}/ckpt_e*.pt"))
last_ckpt = ckpts[-1] if ckpts else None
print("Resuming from", last_ckpt)
assert last_ckpt is not None

subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
    "--backbone", "resnet50",
    "--pretrained",
    "--imagenet-stats",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--resume", last_ckpt,
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "40",
    "--batch-size", "32",
    "--num-workers", "2",
    "--lr", "0.01",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--full-size", "320",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

---

## 6. Peek at history

```python
import json
hist = json.load(open(f"{RUN}/history.json"))
best_idx = max(range(len(hist)), key=lambda i: hist[i]["ema_val_acc"])
print(f"Last epoch: {hist[-1]['epoch']}")
print(f"Best EMA:   {hist[best_idx]['ema_val_acc']:.4f} at ep {hist[best_idx]['epoch']}")
print(f"Best raw:   {max(x['val_acc'] for x in hist):.4f}")
```

---

## 7. Eval + figures — ~5 min

`eval_twostream.py` reads `backbone` / `pretrained` / `imagenet_stats` from the
saved ckpt args; no extra flags needed.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/eval_twostream.py",
    "--ckpt",         f"{RUN}/best.pt",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root",    COMP_DIR,
    "--splits-dir",   f"{WORK}/splits",
    "--out-dir",      f"{RUN}/eval",
    "--history-json", f"{RUN}/history.json",
    "--full-size", "320",
    "--batch-size", "32",
], check=True)
```

Generates in `run9/eval/`:
- `classification_report.txt`, `metrics.json`
- `confusion_matrix.png`, `failures.png`
- `per_driver_accuracy.{png,csv}`
- `training_curves.png`, `attention_grid.png`

---

## 8. Compare to prior runs

```python
import json, pandas as pd, os

def metrics(p):
    m = json.load(open(p))
    return {"accuracy":    m["accuracy"],
            "macro_f1":    m["macro avg"]["f1-score"],
            "weighted_f1": m["weighted avg"]["f1-score"]}

rows = {}
for label, path in [
    ("Run 6 (single stream R18)",   f"{WORK}/run6/eval/metrics.json"),
    ("Run 8 (R18 + pose)",          f"{WORK}/run8/eval/metrics.json"),
    ("Run 9 (R50 ImageNet + pose)", f"{RUN}/eval/metrics.json"),
]:
    if os.path.exists(path):
        rows[label] = metrics(path)

table = pd.DataFrame(rows).T
print(table.to_string())
table.to_csv(f"{WORK}/run9_vs_others.csv")
```

**Expected (actuals from the saved checkpoint):**

| | accuracy | macro_f1 | weighted_f1 |
|---|---|---|---|
| Run 6 | 0.875 | 0.873 | 0.875 |
| Run 8 | 0.939 | 0.939 | 0.939 |
| **Run 9** | **0.949** | **0.948** | **0.949** |

---

## 9. Bundle artifacts for download

```python
import zipfile
from pathlib import Path

OUT = Path(f"{WORK}/artifacts_run9.zip")
OUT.unlink(missing_ok=True)

paths = [
    Path(f"{RUN}/best.pt"),
    Path(f"{RUN}/history.json"),
    *Path(f"{RUN}/eval").iterdir(),
    Path(f"{WORK}/splits/stats.json"),
    Path(f"{WORK}/splits/pose.parquet"),
    Path(f"{WORK}/splits/train.csv"),
    Path(f"{WORK}/splits/val.csv"),
    Path(f"{WORK}/run9_vs_others.csv"),
]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if p.exists():
            z.write(p, p.relative_to(WORK))

print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")
from IPython.display import FileLink, display
display(FileLink(str(OUT)))
```

`best.pt` is ~270 MB (ResNet-50 ~95 M params + pose head). For HuggingFace
Spaces (<50 MB without LFS), save EMA-only:

```python
import torch
ck = torch.load(f"{RUN}/best.pt", map_location="cpu", weights_only=False)
torch.save({"ema": ck["ema"], "args": ck["args"], "pose_fusion": True},
           f"{RUN}/best_demo.pt")
# best_demo.pt ~95 MB — needs LFS for HF Spaces
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--pose-fusion requires --pose-parquet PATH` | flag missing | add `--pose-parquet /kaggle/working/splits/pose.parquet` |
| `AttributeError: 'mediapipe' has no attribute 'solutions'` | newer mediapipe lazy-load broken on Kaggle | `pip install mediapipe==0.10.13` |
| OOM at batch 32 / 320 px | ResNet-50 uses ~2× VRAM vs R18 | `--batch-size 24` or `--full-size 288` |
| Host RAM > 10 GB by ep 3 | DP + worker queue churn | restart kernel, relaunch without `--data-parallel` (single T4, +60% wall time) |
| Val acc < 0.60 by ep 5 | Pretrained weights not loading | verify `--pretrained` flag present; check ckpt `args["pretrained"]==True` |
| `size mismatch` loading ckpt | Run 8 ckpt loaded into Run 9 code (different backbone) | use only `run9/best.pt` with this recipe |
| `attention_grid.png` empty | `use_cbam=False` in ckpt args | re-train without `--no-cbam` |
| Pose detection rate < 60% | unusual lighting / occlusion | lower `min_detection_confidence` in `extract_pose.py` (currently 0.3) |
