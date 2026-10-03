"""
dataset.py - PyTorch Dataset and DataLoader utilities.

Classes
-------
AVWindowDataset
    Loads pre-extracted audio clips and lip-ROI video segments,
    runs frozen encoders on-the-fly, and returns feature tensors.

Functions
---------
collate_varlen
    Custom collate function that pads variable-length audio sequences.

build_dataloaders
    Constructs train / val / test DataLoaders with balanced sampling.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..preprocessing.audio import load_audio_tensor
from ..preprocessing.video import load_video_roi_tensor


class AVWindowDataset(Dataset):
    """
    Audio-Visual Window Dataset.

    Each sample is a pre-processed (audio .wav + lip-ROI .npz) window pair.
    Features are extracted on-the-fly using frozen backbone encoders.

    Parameters
    ----------
    manifest_df            : DataFrame with columns
                             ``audio_path``, ``crop_path``, ``label``.
    wav2vec2_model         : Frozen Wav2Vec2Model on the target device.
    wav2vec2_feat_extractor: Matching Wav2Vec2FeatureExtractor.
    mobilenet              : Frozen MobileNetV3-Small (classifier removed).
    normalize              : ImageNet normalisation transform.
    audio_sr               : Audio sample rate expected by the feature extractor.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        wav2vec2_model,
        wav2vec2_feat_extractor,
        mobilenet,
        normalize,
        audio_sr: int = 16_000,
    ) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.wav2vec2_model         = wav2vec2_model
        self.wav2vec2_feat_extractor = wav2vec2_feat_extractor
        self.mobilenet              = mobilenet
        self.normalize              = normalize
        self.audio_sr               = audio_sr

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row   = self.df.iloc[idx]
        label = int(row["label"])

        # ── Audio ───────────────────────────────────────────────────
        audio_tensor = load_audio_tensor(Path(row["audio_path"]))
        audio_feats  = self._extract_audio_features(audio_tensor)

        # ── Visual ──────────────────────────────────────────────────
        video_tensor  = load_video_roi_tensor(Path(row["crop_path"]))
        visual_feats  = self._extract_visual_features(video_tensor)

        return {
            "audio_features":  audio_feats,   # (L, 768)
            "visual_features": visual_feats,  # (T, 576)
            "labels":          torch.tensor(label, dtype=torch.float),
            "id":              str(row.get("id", idx)),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_audio_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        device = next(self.wav2vec2_model.parameters()).device
        inputs = self.wav2vec2_feat_extractor(
            audio_tensor.numpy(),
            sampling_rate=self.audio_sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)
        outputs = self.wav2vec2_model(input_values)
        return outputs.last_hidden_state.squeeze(0).cpu()  # (L, 768)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_visual_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        device = next(self.mobilenet.parameters()).device
        frames_norm = torch.stack(
            [self.normalize(f) for f in video_tensor], dim=0
        ).to(device)
        feats = self.mobilenet(frames_norm)           # (T, 576)
        return feats.cpu()


# ──────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────

def collate_varlen(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Pad variable-length audio sequences to the batch maximum.

    Visual features are also padded to the batch-max visual length.
    Returns a dict with keys:
    ``audio_features``, ``audio_mask``, ``visual_features``, ``labels``, ``id``.
    """
    audio_feats  = [item["audio_features"]  for item in batch]
    visual_feats = [item["visual_features"] for item in batch]
    labels       = [item["labels"]          for item in batch]
    ids          = [item["id"]              for item in batch]

    # Pad audio
    max_aud = max(f.shape[0] for f in audio_feats)
    audio_padded = torch.zeros(len(batch), max_aud, audio_feats[0].shape[1])
    audio_mask   = torch.zeros(len(batch), max_aud)
    for i, f in enumerate(audio_feats):
        audio_padded[i, :f.shape[0]] = f
        audio_mask  [i, :f.shape[0]] = 1

    # Pad visual
    max_vis = max(f.shape[0] for f in visual_feats)
    visual_padded = torch.zeros(len(batch), max_vis, visual_feats[0].shape[1])
    for i, f in enumerate(visual_feats):
        visual_padded[i, :f.shape[0]] = f

    return {
        "audio_features":  audio_padded,
        "audio_mask":      audio_mask,
        "visual_features": visual_padded,
        "labels":          torch.stack(labels),
        "id":              ids,
    }


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    wav2vec2_model,
    wav2vec2_feat_extractor,
    mobilenet,
    normalize,
    batch_size: int = 8,
    audio_sr:   int = 16_000,
    num_workers: int = 0,   # must be 0 when encoders use CUDA
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build class-balanced train DataLoader and plain val / test DataLoaders.

    Parameters
    ----------
    train_df, val_df, test_df : Pre-processed window manifests.
    *                         : See :class:`AVWindowDataset` for encoder args.
    batch_size                : Mini-batch size.
    audio_sr                  : Audio sample rate.
    num_workers               : Worker processes (must be 0 for CUDA encoders).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    common_kw = dict(
        wav2vec2_model          = wav2vec2_model,
        wav2vec2_feat_extractor = wav2vec2_feat_extractor,
        mobilenet               = mobilenet,
        normalize               = normalize,
        audio_sr                = audio_sr,
    )
    train_dataset = AVWindowDataset(train_df, **common_kw)
    val_dataset   = AVWindowDataset(val_df,   **common_kw)
    test_dataset  = AVWindowDataset(test_df,  **common_kw)

    # Weighted sampler for class balance
    labels  = train_df["label"].values
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weight  = {l: 1.0 / c for l, c in zip(unique_labels, counts)}
    sample_weights = np.array([class_weight.get(int(l), 0.0) for l in labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    dl_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_varlen,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler,  **dl_kw)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,    **dl_kw)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False,    **dl_kw)
    return train_loader, val_loader, test_loader
