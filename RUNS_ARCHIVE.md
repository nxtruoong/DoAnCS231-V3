# Runs Archive

Brief record of all runs. Code for runs 1–8 and 10 has been removed; only Run 9 is runnable. See `notebooks/run9.ipynb` for the live recipe.

---

## Run 1 — Dead training

**Config:** ResNet-18 + CBAM, CutMix, LR 0.1 (no warmup), batch 128, 40 ep.  
**Result:** Loss locked at 2.3026 (ln 10 = random). Accuracy stuck at 10%.  
**Root cause:** Kaiming init on FC(512→10) + LR 0.1 → logit explosion → gradient overflow.

---

## Run 2 — Smoke test

**Config:** No CBAM, no CutMix, no EMA, 3 ep.  
**Result:** Best val acc **0.619**. EMA dead (wrong decay for few steps).  
**Purpose:** Confirmed loss/data pipeline working; exposed EMA bug.

---

## Run 3 — Full aug + broken EMA

**Config:** CBAM + CutMix + EMA 0.999, warmup added, 10 ep.  
**Result:** Best raw val acc **0.743**. EMA stuck at 10% (bug: decay applied to BN buffers too).  
**Fix:** Split params from buffers in EMA.update.

---

## Run 4 — EMA fix verified

**Config:** Same as Run 3, EMA fixed, 25 ep.  
**Result:** Best raw **0.797**, best EMA **0.589** (EMA still lagging — decay 0.999 too slow).  
**Fix:** Lower EMA decay to 0.99.

---

## Run 5 — TrivialAugment + 320 px input

**Config:** ResNet-18 + CBAM, TrivialAugment, 320×320, EMA 0.99, early stop, batch 128, up to 80 ep.  
**Result:** Best raw **0.833** (ep 31), best EMA **0.843** (ep 30). Stopped ep 38.  
**Note:** +5 pp over Run 4. First run with a healthy EMA.

---

## Run 6 — 384 px + tighter crop (headline single-stream)

**Config:** ResNet-18 + CBAM, 384×384, CutMix p halved, same schedule.  
**Result:** Accuracy **0.875**, macro F1 **0.873**, weighted F1 **0.875**.  
**Notes:** +3 pp over Run 5 EMA. c3 F1 0.77 → 0.92 (bigger input + crop). Ceiling: c0/c9 passive classes hard without pose.

---

## Run 7 — Two-stream (full 384 + top-crop face) — REGRESSED

**Config:** Two ResNet-18+CBAM backbones; full 384 + top-50% crop 224; concat → MLP → 10. ~22.8M params.  
**Result:** Accuracy **0.751**, macro F1 **0.748**. −12 pp vs Run 6.  
**Root cause:** Second stream saw only upper body → c4 phone-left (foreground arm) destroyed; no face-stream gain on passive classes. Killed at ep 47.

---

## Run 8 — Pose-fusion R18+CBAM + MediaPipe 36-d (prior headline)

**Config:** Single ResNet-18+CBAM (full 384) + 36-d MediaPipe Pose vector (head, wrists, elbows, fingers, hips + derived signals). Pose MLP FC(36→128→128), concat 640 → FC(640→256→10). No CutMix.  
**Result:** Accuracy **0.939**, macro F1 **0.939**, weighted F1 **0.939**.  
**Notes:** +6.4 pp over Run 6. c3 F1 0.92→0.95, c9 0.68→0.81. Pose gives gaze + arm-position signal the CNN cannot infer from pixels alone.

---

## Run 9 — ResNet-50 ImageNet + CBAM + pose-fusion (CURRENT BEST)

**Config:** ResNet-50 (ImageNet1K_V2 pretrained) + CBAM after each of 4 stages; same 36-d MediaPipe pose vector as Run 8. LR 0.01, warmup 2 ep, 320×320, batch 32, T4×2.  
**Result:** Accuracy **0.949**, macro F1 **0.948**, weighted F1 **0.949**.  
**Notes:** +1.0 pp over Run 8. Deeper backbone (2048-d vs 512-d) + ImageNet transfer. Weakest class: c9 talk-passenger F1 0.797.  
**Runnable:** see `notebooks/run9.ipynb`.

---

## Run 10 — ResNet-50 ImageNet + CBAM, image-only ablation

**Config:** ResNet-50 + CBAM, no pose, no EMA, no CutMix. Left 80% crop to remove empty driver-side background. 320×320, batch 32.  
**Purpose:** Isolate backbone upgrade contribution from pose-fusion contribution.  
**Status:** Ablation design recorded; code removed. See `log.md` Run 10 entry.
