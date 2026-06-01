# Driver Distraction Classification (ResNet-50 + CBAM + MediaPipe Pose Fusion)

End-term project for Computer Vision (UIT). Classifies the 10 State Farm
distracted-driver classes using a ResNet-50 (ImageNet pretrained) with CBAM
attention blocks fused with a 36-d MediaPipe Pose vector. Evaluated with a
**subject-wise split** so val accuracy reflects real generalization, not
driver-identity memorization.

See [`log.md`](log.md) for full per-run training history and [`RUNS_ARCHIVE.md`](RUNS_ARCHIVE.md)
for a brief summary of all runs.

## Result

**Accuracy 0.949 · Macro F1 0.948 · Weighted F1 0.949** on subject-wise val
(4 524 images, 5 held-out drivers).

| class | F1 |
|---|---:|
| c0 safe driving | 0.839 |
| c1 texting right | 0.997 |
| c2 phone right | 0.996 |
| c3 texting left | 0.985 |
| c4 phone left | 0.987 |
| c5 radio | 0.987 |
| c6 drinking | 0.991 |
| c7 reach behind | 0.966 |
| c8 hair/makeup | 0.937 |
| c9 talk passenger | 0.797 |

## Architecture

| | |
|---|---|
| Backbone | ResNet-50, ImageNet1K_V2 pretrained, CBAM (CAM + SAM) after each of layer1–4 |
| Pose stream | 36-d MediaPipe Pose vector → MLP(36→128→128) |
| Fusion | concat(2048-d CNN features, 128-d pose) → FC(2176→256→10) |
| Split | Subject-wise; held-out: `p022, p035, p047, p056, p075` |
| Augmentation | RandomResizedCrop + TrivialAugmentWide + Normalize + RandomErasing(p=0.25). **No HFlip** (left/right class asymmetry). No CutMix (incompatible with pose-fusion path) |
| Normalization | ImageNet mean/std (matches pretrained weights) |
| Optimizer | SGD (Nesterov), momentum=0.9, weight_decay=5e-4 |
| LR schedule | Linear warmup 2 ep → Cosine 0.01 → 0 |
| Loss | CrossEntropy + label smoothing 0.1 |
| EMA decay | 0.99 |
| Batch / input size | 32 / 320×320 |
| Max epochs / early stop | 40 / patience 8 |
| Training time | ~2–2.5 hr on Kaggle T4×2 |

## Architecture history

| Run | Architecture | Macro F1 |
|---|---|---:|
| Run 5 | ResNet-18 + CBAM from scratch, 320 px | 0.843 |
| Run 6 | ResNet-18 + CBAM from scratch, 384 px | 0.873 |
| Run 7 | Two-stream (full + top-crop), 384 px | 0.748 ↓ regressed |
| Run 8 | ResNet-18 + CBAM + MediaPipe pose, 384 px | 0.939 |
| **Run 9** | **ResNet-50 ImageNet + CBAM + MediaPipe pose, 320 px** | **0.948** |

## Repo layout

```
.
├── data_prep.py           # subject-wise split + dataset RGB stats
├── augment.py             # augmentation pipeline, dataset class, CutMix
├── augment_twostream.py   # PoseFusionDataset, pose lookup, TwoStreamDataset
├── model.py               # ResNet-18 + CBAM (used by model_twostream)
├── model_resnet50.py      # ResNet-50 + CBAM builder
├── model_twostream.py     # PoseFusionCBAM + build_posefusion (Run 9 model)
├── train_twostream.py     # training loop — use --pose-fusion --backbone resnet50 --pretrained
├── eval_twostream.py      # eval + figures, auto-dispatches on ckpt args
├── eval.py                # shared eval helpers (imported by eval_twostream)
├── extract_pose.py        # MediaPipe Pose precompute → splits/pose.parquet
├── notebooks/
│   └── run9.ipynb         # end-to-end Kaggle notebook (canonical entry point)
├── log.md                 # per-run training log + per-class diagnosis
├── RUN9_HOWTO.md          # step-by-step Kaggle T4×2 runbook for Run 9
├── RUNS_ARCHIVE.md        # brief results for Runs 1–10
├── requirements.txt
└── README.md
```

## Quickstart (Kaggle T4×2)

The canonical entry point is **`notebooks/run9.ipynb`**.

1. Open a new Kaggle notebook. Attach the **State Farm Distracted Driver Detection**
   competition dataset. Set Accelerator = GPU T4×2, Internet = On.
2. Make scripts importable — either upload this repo as a Kaggle dataset at
   `/kaggle/input/driver-distraction-cbam`, or clone it:
   ```python
   !git clone https://github.com/nxtruoong/DoAnCS231-V3 /kaggle/working/code
   CODE_DIR = "/kaggle/working/code"
   ```
3. Open `notebooks/run9.ipynb` and run cells top-to-bottom.

For a plain-script alternative see [`RUN9_HOWTO.md`](RUN9_HOWTO.md). Key training command:

```bash
python train_twostream.py \
    --pose-fusion \
    --backbone resnet50 --pretrained --imagenet-stats \
    --pose-parquet /kaggle/working/splits/pose.parquet \
    --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \
    --splits-dir /kaggle/working/splits \
    --out-dir    /kaggle/working/run9 \
    --epochs 40 --batch-size 32 --num-workers 2 \
    --lr 0.01 --warmup-epochs 2 --ema-decay 0.99 \
    --full-size 320 --label-smoothing 0.1 \
    --early-stop-patience 8 --data-parallel
```

## Reproducibility caveat

`torch.backends.cudnn.deterministic=True` and `seed=42` throughout, but CUDA +
cuDNN versions on Kaggle change over time so exact bitwise reproducibility across
sessions is not guaranteed.

## References

**Backbone + transfer learning**
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. arXiv:1512.03385.
  — ResNet architecture (BasicBlock / Bottleneck layout, global average pool, fc head).
- He, K. et al. (2019). *Bag of Tricks for Image Classification with Convolutional Neural Networks.*
  CVPR. arXiv:1812.01187. — ResNet training recipe; basis for LR warmup + cosine schedule used here.

**Attention module**
- Woo, S. et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV. arXiv:1807.06521.
  — Channel + spatial attention block inserted after each ResNet stage. Spatial attention map
  from `layer4` produces the eval heatmap overlay.

**Pose estimation**
- Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.*
  arXiv:1906.08172. — MediaPipe framework used for on-device pose landmark extraction.
- Bazarevsky, V. et al. (2020). *BlazePose: On-device Real-time Body Pose Tracking.*
  arXiv:2006.10204. — BlazePose model underlying `mediapipe.solutions.pose`; produces the
  33 landmarks from which the 36-d feature vector is engineered.

**Regularization + augmentation**
- Zhong, Z. et al. (2020). *Random Erasing Data Augmentation.* AAAI. arXiv:1708.04896.
- Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer Vision.*
  CVPR. arXiv:1512.00567. — Label smoothing.

**Optimization**
- Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.*
  ICLR. arXiv:1608.03983. — Cosine LR schedule.
- Polyak, B. & Juditsky, A. (1992). *Acceleration of Stochastic Approximation by Averaging.*
  SIAM J. Control Optim. — Theoretical basis for EMA weight averaging at eval time.

**Related work on State Farm**
- Eraqi, H. M. et al. (2019). *Driver Distraction Identification with an Ensemble of
  Convolutional Neural Networks.* J. Adv. Transp. arXiv:1901.09097.
- Masood, S. et al. (2018). *Detecting Distracted Driver Using Convolutional Neural Network.*
  CVPR Workshops.
