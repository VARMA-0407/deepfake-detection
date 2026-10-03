# Deepfake Detection via Audio-Visual Representation Training (ART)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Linux-lightgrey)

**A two-stage audio-visual deepfake detector that learns cross-modal synchrony through contrastive pre-training, then classifies fake videos with a lightweight Focal-Loss MLP.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Training Pipeline](#training-pipeline)
- [Inference & Evaluation](#inference--evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

Deepfake videos manipulate either the visual (face) or audio (voice) stream — but rarely both perfectly in sync. **ART (Audio-visual Representation Training)** exploits this by:

1. **Stage 1 — Contrastive Pre-training**: Learning a joint audio-visual embedding space where real audio-lip pairs are pulled together (InfoNCE loss) while fake pairs are pushed apart.
2. **Stage 2 — Supervised Fine-tuning**: A small Fusion MLP is trained on top of the frozen ART embeddings to classify windows as *real* (0) or *fake* (1).
3. **Video-level Aggregation**: Multiple overlapping 1-second windows are aggregated (mean / max / top-k / alignment-weighted) to produce a final per-video score.

---

## Architecture

```
Video Input
    │
    ├─── Audio stream ──► FFmpeg slice ──► Wav2Vec2 (frozen) ──► BiLSTM ──► audio_global (256-d)
    │                                                                  └──► ProjectionHead ──► a_proj (128-d, L2-norm)
    │
    └─── Video stream ──► MediaPipe lip-ROI ──► MobileNetV3 (frozen) ──► TCN ──► visual_global (256-d)
                                                                               └──► ProjectionHead ──► v_proj (128-d, L2-norm)

Stage 1 (ART):  InfoNCE(a_proj, v_proj)

Stage 2 (Det):  FusionMLP([audio_global | visual_global | a_proj]) ──► logit ──► sigmoid ──► fake probability
```

### Key components

| Component | Description | Parameters |
|---|---|---|
| `Wav2Vec2-base` | Audio backbone (frozen) | 94 M (frozen) |
| `MobileNetV3-Small` | Visual backbone (frozen) | 2.5 M (frozen) |
| `AudioTemporalHead` | 2-layer BiLSTM → 256-d | ~3.5 K |
| `VisualTemporalHead` | 3-layer TCN → 256-d | ~590 K |
| `ProjectionHead` | MLP → 128-d L2-normalised | ~66 K each |
| `FusionDetectorMLP` | 640→256→128→1 MLP | ~100 K |

---

## Project Structure

```
deepfake-detection/
├── deepfake_detection/               # Main Python package
│   ├── __init__.py                   # Public API
│   ├── config.py                     # All hyperparameters & paths
│   ├── dataset.py                    # AVWindowDataset + DataLoaders
│   ├── losses.py                     # InfoNCE contrastive loss
│   ├── train_art.py                  # Stage 1: ART pre-training
│   ├── train_detector.py             # Stage 2: Detector fine-tuning
│   ├── inference.py                  # Window inference + video aggregation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio.py                  # FFmpeg slicing, audio tensor loading
│   │   └── video.py                  # MediaPipe lip-ROI extraction
│   └── models/
│       ├── __init__.py
│       ├── encoders.py               # Frozen Wav2Vec2 + MobileNetV3
│       ├── heads.py                  # BiLSTM, TCN, ProjectionHead
│       └── detector.py               # FusionDetectorMLP + FocalLoss
├── NLP_FINAL_PROJECT (1).ipynb       # Original Colab notebook
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: A100 / T4 on Google Colab)
- `ffmpeg` and `mediainfo` system packages

```bash
# System dependencies (Ubuntu / Colab)
apt-get update -qq && apt-get install -y ffmpeg mediainfo

# Python dependencies
pip install -r requirements.txt
```

### Clone

```bash
git clone https://github.com/VARMA-0407/deepfake-detection.git
cd deepfake-detection
```

---

## Quick Start

### Google Colab

Open the original notebook for an end-to-end walkthrough:  
📔 [`NLP_FINAL_PROJECT (1).ipynb`](NLP_FINAL_PROJECT%20(1).ipynb)

### Modular scripts

```python
# 1. Set your data directory
import os
os.environ["DEEPFAKE_BASE_DIR"] = "/content/deepfake_detection"

# 2. Stage 1 – ART contrastive pre-training
from deepfake_detection.train_art import run_art_pretraining
run_art_pretraining()

# 3. Stage 2 – Detector fine-tuning
from deepfake_detection.train_detector import run_detector_training
run_detector_training()

# 4. Inference
from deepfake_detection.inference import load_models_for_inference, predict_windows
audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector = \
    load_models_for_inference(device)
window_results = predict_windows(test_loader, audio_temp, visual_temp,
                                 audio_proj_h, visual_proj_h, detector, device)
```

---

## Module Reference

### `config.py`
Central configuration. Override via environment variables or edit directly.

| Variable | Default | Description |
|---|---|---|
| `WINDOW_SEC` | `1.0` | Window duration (seconds) |
| `WINDOW_STRIDE_SEC` | `0.5` | Stride between windows |
| `AUDIO_SR` | `16000` | Audio sample rate |
| `CROP_SIZE` | `112` | Lip-ROI spatial size (px) |
| `TARGET_FPS` | `25` | Video frame rate |
| `ART_EPOCHS` | `10` | Contrastive pre-training epochs |
| `ART_LR` | `1e-4` | ART learning rate |
| `ART_TEMP` | `0.07` | InfoNCE temperature |
| `DETECTOR_EPOCHS` | `8` | Detector fine-tuning epochs |
| `BATCH_SIZE` | `8` | Mini-batch size |

### `preprocessing/audio.py`
| Function | Description |
|---|---|
| `slice_audio_ffmpeg(video_path, start, end, out_wav, sr)` | FFmpeg-based mono audio extraction |
| `load_audio_tensor(audio_path, target_sr, target_len)` | Load WAV as fixed-length tensor |

### `preprocessing/video.py`
| Function | Description |
|---|---|
| `extract_lip_roi_window(video_path, start, end, out_npz, ...)` | MediaPipe face detection → lip crop → save .npz |
| `load_video_roi_tensor(npz_path)` | Load saved ROI tensor |

### `models/heads.py`
| Class | Input → Output | Purpose |
|---|---|---|
| `AudioTemporalHead` | `(B,L,768) → (B,256)` | BiLSTM over Wav2Vec2 features |
| `VisualTemporalHead` | `(B,T,576) → (B,256)` | TCN over MobileNetV3 features |
| `ProjectionHead` | `(B,256) → (B,128)` | L2-normalised contrastive projection |

### `losses.py`
| Function | Description |
|---|---|
| `info_nce_loss(audio_proj, visual_proj, temperature)` | Symmetric InfoNCE (CLIP-style) loss |

### `inference.py`
| Function | Description |
|---|---|
| `predict_windows(dataloader, ...)` | Batch inference → list of window dicts |
| `aggregate_windows_to_video(results, method)` | `"mean"` / `"max"` / `"topk"` / `"align_weighted"` |
| `evaluate_video_level(win_df, threshold, method)` | Compute AUC / F1 / Balanced Accuracy |

---

## Training Pipeline

```
Raw videos (real + fake)
        │
        ▼
  Window Creation        (1-sec windows, 0.5-sec stride)
        │
        ▼
  Preprocessing          (FFmpeg audio + MediaPipe lip crops)
        │
        ▼
  Stage 1: ART           (InfoNCE, 10 epochs, lr=1e-4)
        │  Saves: checkpoints/art_epoch{N}.pt
        ▼
  Stage 2: Detector      (FocalLoss, 8 epochs, early stopping)
        │  Saves: checkpoints/best_detector.pt
        ▼
  Inference + Aggregation
        │
        ▼
  Video-level AUC / F1 / Balanced Accuracy
```

---

## Inference & Evaluation

```python
import pandas as pd
from deepfake_detection.inference import evaluate_video_level

win_df = pd.read_csv("checkpoints/window_predictions.csv")
evaluate_video_level(win_df, threshold=0.5, method="align_weighted")
```

**Aggregation methods compared:**

| Method | Description | Best for |
|---|---|---|
| `mean` | Average all window probs | Balanced videos |
| `max` | Most suspicious window | Videos with short edits |
| `topk` | Average top 33 % windows | Noise-robust |
| `align_weighted` | Weight by audio-visual mis-alignment | Subtle manipulations |

---

## Configuration

Set `DEEPFAKE_BASE_DIR` to point to your data directory:

```bash
export DEEPFAKE_BASE_DIR=/path/to/your/data
```

Expected directory layout under `DEEPFAKE_BASE_DIR`:

```
data/          ← raw video files (.mp4)
splits/        ← CSV manifests (train/val/test_preprocessed.csv)
audio_clips/   ← .wav window clips
lip_crops/     ← .npz lip-ROI tensors
checkpoints/   ← model checkpoints
```

---

## Results

> Results shown on a balanced subset (50 windows/class/split) in testing mode.

| Metric | Value |
|---|---|
| AUC-ROC | ~0.82 |
| F1-Score | ~0.76 |
| Balanced Accuracy | ~0.79 |

*Full-dataset results may differ. Training on the complete FaceForensics++ or similar benchmark is recommended for production use.*

---

## Requirements

```
torch>=2.0
torchvision>=0.15
torchaudio>=2.0
transformers>=4.30
mediapipe>=0.10
opencv-contrib-python
scikit-learn
pandas
numpy<2
tqdm
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{varma2024deepfake,
  title   = {Deepfake Detection via Audio-Visual Representation Training},
  author  = {VARMA-0407},
  year    = {2024},
  url     = {https://github.com/VARMA-0407/deepfake-detection}
}
```

---

## License

MIT License © 2024 VARMA-0407
