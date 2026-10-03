"""
train_art.py - ART (Audio-visual Representation Training) contrastive pre-training.

Run this script to pre-train the temporal and projection heads with InfoNCE loss
before supervised detector fine-tuning.

Usage
-----
    python -m deepfake_detection.train_art
"""

from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from .config import (
    ART_EPOCHS, ART_LR, ART_TEMP, BATCH_SIZE,
    CKPT_DIR, DEVICE, SPLITS_DIR,
    TEMPORAL_DIM, AUDIO_FEAT_DIM, VISUAL_FEAT_DIM, PROJ_DIM,
)
from .dataset import build_dataloaders
from .losses import info_nce_loss
from .models.encoders import load_frozen_encoders
from .models.heads import AudioTemporalHead, VisualTemporalHead, ProjectionHead

import pandas as pd


def run_art_pretraining() -> None:
    """Execute the full ART contrastive pre-training loop."""
    # ── Data ────────────────────────────────────────────────────────────
    train_df = pd.read_csv(SPLITS_DIR / "train_preprocessed.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_preprocessed.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_preprocessed.csv")

    wav2vec2, feat_extractor, mobilenet, normalize = load_frozen_encoders(DEVICE)

    train_loader, val_loader, _ = build_dataloaders(
        train_df, val_df, test_df,
        wav2vec2, feat_extractor, mobilenet, normalize,
        batch_size=BATCH_SIZE,
    )

    # ── Models ──────────────────────────────────────────────────────────
    audio_temp  = AudioTemporalHead(AUDIO_FEAT_DIM,  256, TEMPORAL_DIM).to(DEVICE)
    visual_temp = VisualTemporalHead(VISUAL_FEAT_DIM, 256, TEMPORAL_DIM).to(DEVICE)
    audio_proj  = ProjectionHead(TEMPORAL_DIM, PROJ_DIM).to(DEVICE)
    visual_proj = ProjectionHead(TEMPORAL_DIM, PROJ_DIM).to(DEVICE)

    params = (
        list(audio_temp.parameters())  +
        list(visual_temp.parameters()) +
        list(audio_proj.parameters())  +
        list(visual_proj.parameters())
    )
    optimizer = Adam(params, lr=ART_LR, weight_decay=1e-6)

    print(f"\n{'='*60}")
    print("ART CONTRASTIVE PRE-TRAINING")
    print(f"{'='*60}")
    print(f"Epochs : {ART_EPOCHS}")
    print(f"LR     : {ART_LR}")
    print(f"Temp   : {ART_TEMP}")
    print(f"Params : {sum(p.numel() for p in params if p.requires_grad):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, ART_EPOCHS + 1):
        for m in [audio_temp, visual_temp, audio_proj, visual_proj]:
            m.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"ART Epoch {epoch}/{ART_EPOCHS}")

        for batch in pbar:
            audio_feats  = batch["audio_features"].to(DEVICE)
            audio_mask   = batch["audio_mask"].to(DEVICE)
            visual_feats = batch["visual_features"].to(DEVICE)

            a_global = audio_temp(audio_feats, mask=audio_mask)
            v_global = visual_temp(visual_feats)
            a_proj   = audio_proj(a_global)
            v_proj   = visual_proj(v_global)

            loss = info_nce_loss(a_proj, v_proj, temperature=ART_TEMP)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"  Epoch {epoch} — avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"art_epoch{epoch}.pt"
        torch.save(
            {
                "epoch":            epoch,
                "audio_temp_state":  audio_temp.state_dict(),
                "visual_temp_state": visual_temp.state_dict(),
                "audio_proj_state":  audio_proj.state_dict(),
                "visual_proj_state": visual_proj.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"  Checkpoint saved: {ckpt_path}")

    print("\nART pre-training complete.")


if __name__ == "__main__":
    run_art_pretraining()
