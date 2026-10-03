"""
config.py - Global configuration for Deepfake Detection system.
All hyperparameters, paths, and runtime settings are defined here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR   = Path(os.environ.get("DEEPFAKE_BASE_DIR", "/content/deepfake_detection"))
DATA_DIR   = BASE_DIR / "data"
SPLITS_DIR = BASE_DIR / "splits"
AUDIO_DIR  = BASE_DIR / "audio_clips"
CROPS_DIR  = BASE_DIR / "lip_crops"
CKPT_DIR   = BASE_DIR / "checkpoints"

for d in [DATA_DIR, SPLITS_DIR, AUDIO_DIR, CROPS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Dataset / Windowing
# ──────────────────────────────────────────────
WINDOW_SEC          = 1.0    # window duration in seconds
WINDOW_STRIDE_SEC   = 0.5    # stride between consecutive windows
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15

# Testing / debug mode (limits samples per split)
TESTING_MODE             = True
TEST_SAMPLES_PER_SPLIT   = 50   # windows per class per split in testing mode

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
AUDIO_SR    = 16_000   # target sampling rate (Hz)
CROP_SIZE   = 112      # lip-ROI spatial size (pixels)
TARGET_FPS  = 25       # target video frame rate
LIP_PADDING = 0.35     # relative padding around lip region
SMOOTH_WIN  = 5        # smoothing window for bounding boxes

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
AUDIO_FEAT_DIM   = 768   # Wav2Vec2-base hidden size
VISUAL_FEAT_DIM  = 576   # MobileNetV3-Small feature size
TEMPORAL_DIM     = 256   # BiLSTM / TCN output dim
PROJ_DIM         = 128   # contrastive projection dim

# ──────────────────────────────────────────────
# ART (Audio-visual Representation Training)
# ──────────────────────────────────────────────
ART_EPOCHS = 10
ART_LR     = 1e-4
ART_TEMP   = 0.07    # InfoNCE temperature

# ──────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────
DETECTOR_EPOCHS = 8
DETECTOR_LR     = 1e-4
PATIENCE        = 5
BATCH_SIZE      = 8

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
