# deepfake_detection package
"""
Deepfake Detection via Audio-Visual Representation Training (ART).

Package layout
--------------
config.py           : Global hyperparameters and paths.
dataset.py          : AVWindowDataset + DataLoader factory.
losses.py           : InfoNCE contrastive loss.
train_art.py        : ART contrastive pre-training script.
train_detector.py   : Supervised detector fine-tuning script.
inference.py        : Window-level inference + video-level aggregation.

preprocessing/
    audio.py        : FFmpeg audio slicing and tensor loading.
    video.py        : MediaPipe lip-ROI extraction and tensor loading.

models/
    encoders.py     : Frozen Wav2Vec2 + MobileNetV3 backbones.
    heads.py        : AudioTemporalHead (BiLSTM), VisualTemporalHead (TCN),
                      ProjectionHead (MLP).
    detector.py     : FusionDetectorMLP + FocalLoss.
"""

from .config import DEVICE
from .models  import (
    load_frozen_encoders,
    AudioTemporalHead,
    VisualTemporalHead,
    ProjectionHead,
    FusionDetectorMLP,
    FocalLoss,
)
from .dataset   import AVWindowDataset, collate_varlen, build_dataloaders
from .losses    import info_nce_loss
from .inference import predict_windows, aggregate_windows_to_video, evaluate_video_level

__all__ = [
    "DEVICE",
    "load_frozen_encoders",
    "AudioTemporalHead",
    "VisualTemporalHead",
    "ProjectionHead",
    "FusionDetectorMLP",
    "FocalLoss",
    "AVWindowDataset",
    "collate_varlen",
    "build_dataloaders",
    "info_nce_loss",
    "predict_windows",
    "aggregate_windows_to_video",
    "evaluate_video_level",
]
