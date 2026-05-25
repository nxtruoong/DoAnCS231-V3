# RUN10_HOWTO — ResNet-50 + ImageNet + CBAM (image-only, cropped)

Single-stream Kaggle T4×2 recipe. No MediaPipe, no pose, no second
stream, **no EMA, no CutMix**. Inputs cropped to the left 80% of each
frame so the model stops attending to the empty area behind the driver
seat.

Goal: isolate the contribution of a deeper, ImageNet-pretrained
backbone with CBAM, against Run 6 (ResNet-18 + CBAM from scratch) and
Run 8 / 9 (pose / dual-stream variants).

Entry points are `train_run10.py` and `eval_run10.py` (separate from
`train.py` / `eval.py`, which still serve Run 6 / 8 / 9 with EMA + CutMix).
For design context see `log.md` (Run 10 entry). This file is the runbook.

---

## 0. Prerequisites

- Subject-wise splits already produced (`splits/train.csv`,
  `splits/val.csv`). `stats.json` is **not required** for Run 10 — we
  use ImageNet stats by default. Only needed if you pass `--dataset-stats`.
- Code dataset attached at `/kaggle/input/driver-distraction-cbam` or
  cloned to `/kaggle/working/code`. Must contain:
  - `train_run10.py`, `eval_run10.py`
  - `model.py`, `model_resnet50.py`
  - `augment.py` (with `StateFarmCropDataset`)

GitHub mirror:
```python
!rm -rf /kaggle/working/code
!git clone https://github.com/nxtruoong/DoAnCS231 /kaggle/working/code
CODE_DIR = "/kaggle/working/code"
```

---

## 1. Paths (Cell 1)

```python
import os, sys
COMP_DIR = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
WORK     = "/kaggle/working"
RUN      = f"{WORK}/run10"

assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(CODE_DIR + "/train_run10.py"),       "Run 10 code missing"

sys.path.insert(0, CODE_DIR)
print("OK. GPU count:", __import__("torch").cuda.device_count())
```

---

## 2. Smoke test (Cell 2, ~5–7 min)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_run10.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run10_smoke",
    "--backbone", "resnet50",
    "--pretrained",
    "--epochs", "2",
    "--batch-size", "32",
    "--num-workers", "4",
    "--lr", "0.01",
    "--warmup-epochs", "1",
    "--img-size", "320",
    "--crop-left-frac", "0.8",
    "--data-parallel",
], check=True)
```

Expect: 2 epochs complete, no OOM. Val acc 0.45–0.75 by ep 2 (ImageNet
init + crop). If OOM → drop `--batch-size` to 24 or `--img-size` to 288.

---

## 3. Full Run 10 training (Cell 3, ~2–2.5 hr)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_run10.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--backbone", "resnet50",
    "--pretrained",
    "--epochs", "30",
    "--batch-size", "32",
    "--num-workers", "4",
    "--lr", "0.01",
    "--warmup-epochs", "2",
    "--weight-decay", "1e-4",
    "--label-smoothing", "0.1",
    "--img-size", "320",
    "--crop-left-frac", "0.8",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.005",
    "--ckpt-every", "5",
    "--data-parallel",
], check=True)
```

**Why these hyperparameters differ from Run 6:**
- `--lr 0.01` (vs Run 6 `0.1`): pretrained weights need gentler LR.
- `--weight-decay 1e-4` (vs `5e-4`): standard for fine-tuning.
- `--epochs 30` (vs `50`): pretrained converges faster.
- ImageNet stats are the default (pretrained backbone expects them).
- `--img-size 320` (vs 224): higher res helps after cropping out 20%.
- No EMA, no CutMix: simplifies the recipe; the gains from those two
  on Run 6 were marginal (≤0.5 acc) compared to the depth + pretraining
  + crop signal that Run 10 is testing.

**Checkpoints land in** `/kaggle/working/run10/`: `best.pt`,
`ckpt_e05.pt`, ..., `final.pt`. Selection is on raw `val_acc` (no EMA).

**Watch milestones** (Run 10 should beat Run 6 at every checkpoint):

| ep | target val acc | Run 6 actual |
|---:|---:|---:|
| 05 | ≥ 0.70 | ~0.50 |
| 10 | ≥ 0.82 | ~0.78 |
| 20 | ≥ 0.87 | 0.82  |
| 30 | ≥ 0.88 | —     |

If val acc stalls below 0.80 by epoch 10 → halve `--lr` to `0.005`.

---

## 4. Evaluation (Cell 4)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/eval_run10.py",
    "--ckpt",        f"{RUN}/best.pt",
    "--data-root",   COMP_DIR,
    "--splits-dir",  f"{WORK}/splits",
    "--out-dir",     f"{RUN}/eval",
    "--history-json", f"{RUN}/history.json",
    "--batch-size", "64",
    "--num-workers", "4",
    "--img-size", "320",
], check=True)
```

`--crop-left-frac`, `--dataset-stats`, and `--backbone` are auto-detected
from `saved_args`. Override `--crop-left-frac` only when probing a
mismatched eval crop.

Artifacts written to `/kaggle/working/run10/eval/`:
- `metrics.json`, `classification_report.txt`
- `confusion_matrix.png`
- `per_driver_accuracy.{csv,png}`
- `training_curves.png` (train/val only, no EMA traces)
- `attention_grid.png` (SAM overlays from `cbam4`)
- `failures.png`

---

## 5. Ablation knobs

| Want to test | Flag combo |
|---|---|
| No crop (full frame) | `--crop-left-frac 1.0` |
| Tighter crop (left 70%) | `--crop-left-frac 0.7` |
| ResNet-50 from scratch (no ImageNet) | drop `--pretrained`; add `--dataset-stats`; raise `--lr` to `0.1` |
| ResNet-50 + ImageNet, no CBAM | add `--no-cbam` |
| Dataset stats instead of ImageNet | `--dataset-stats` (requires `stats.json`) |
| Same recipe on ResNet-18 | `--backbone resnet18`, drop `--pretrained` |
| Smaller resolution (faster) | `--img-size 224` |

---

## 6. Troubleshooting

- **OOM at batch 32 / 320 px.** Drop `--batch-size` to 24 or 16. Or
  `--img-size 288`. ResNet-50 @ 320 is tight on a single T4.
- **First-epoch val acc < 0.20.** `--lr` too high for pretrained
  weights; halve.
- **Eval crashes loading ckpt.** Run 10 ckpts have no `ema` key.
  `eval_run10.py` loads `ckpt["model"]` directly. If you point
  `eval.py` (the shared script) at a Run 10 ckpt it will fail —
  use `eval_run10.py`.
- **Crop looks wrong.** Inspect with:
  ```python
  from PIL import Image
  im = Image.open("…/c0/img_xxx.jpg").convert("RGB")
  w, h = im.size
  im.crop((0, 0, int(w * 0.8), h)).save("/tmp/crop_check.jpg")
  ```
  Adjust `--crop-left-frac` if the driver's hands clip out on phone-left
  classes.
