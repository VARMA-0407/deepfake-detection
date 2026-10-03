"""
encoders.py - Frozen pre-trained backbone encoders.

load_frozen_encoders() returns:
    wav2vec2_model            : facebook/wav2vec2-base (frozen)
    wav2vec2_feature_extractor: matching feature extractor
    mobilenet                 : MobileNetV3-Small without classifier (frozen)
    normalize                 : torchvision ImageNet normalisation transform
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


def load_frozen_encoders(device: torch.device):
    """
    Load and freeze both backbone encoders.

    Parameters
    ----------
    device : torch.device
        Target device (CPU or CUDA).

    Returns
    -------
    tuple
        ``(wav2vec2_model, wav2vec2_feature_extractor, mobilenet, normalize)``
    """
    # ── Audio encoder ──────────────────────────────────────────────────────
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    wav2vec2_model.eval()
    for p in wav2vec2_model.parameters():
        p.requires_grad = False
    wav2vec2_model = wav2vec2_model.to(device)

    # ── Visual encoder ─────────────────────────────────────────────────────
    mobilenet = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    mobilenet.classifier = nn.Identity()   # strip classification head
    mobilenet.eval()
    for p in mobilenet.parameters():
        p.requires_grad = False
    mobilenet = mobilenet.to(device)

    # ── ImageNet normalisation ─────────────────────────────────────────────
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    print("Frozen encoders loaded:")
    print(
        f"  Wav2Vec2:     "
        f"{sum(p.numel() for p in wav2vec2_model.parameters()) / 1e6:.1f}M params (frozen)"
    )
    print(
        f"  MobileNetV3:  "
        f"{sum(p.numel() for p in mobilenet.parameters()) / 1e6:.1f}M params (frozen)"
    )

    return wav2vec2_model, wav2vec2_feature_extractor, mobilenet, normalize
