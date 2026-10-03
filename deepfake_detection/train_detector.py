"""
train_detector.py - Supervised detector fine-tuning.

Loads the best ART checkpoint, freezes the temporal/projection heads,
then fine-tunes only the FusionDetectorMLP with Focal Loss.

Usage
-----
    python -m deepfake_detection.train_detector
"""

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from tqdm import tqdm

from .config import (
    ART_EPOCHS, BATCH_SIZE, CKPT_DIR, DETECTOR_EPOCHS, DETECTOR_LR,
    DEVICE, FOCAL_ALPHA, FOCAL_GAMMA, PATIENCE, SPLITS_DIR,
    TEMPORAL_DIM, AUDIO_FEAT_DIM, VISUAL_FEAT_DIM, PROJ_DIM,
)
from .dataset import build_dataloaders
from .models.detector import FocalLoss, FusionDetectorMLP
from .models.encoders import load_frozen_encoders
from .models.heads import AudioTemporalHead, VisualTemporalHead, ProjectionHead


# ──────────────────────────────────────────────
# Validation helper
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(val_loader, audio_temp, visual_temp, audio_proj_h, visual_proj_h,
             detector, loss_fn, device):
    """Evaluate on the validation set and return balanced metrics."""
    for m in [audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector]:
        m.eval()

    all_labels, all_probs, all_preds = [], [], []
    total_loss = 0.0

    for batch in val_loader:
        af  = batch["audio_features"].to(device)
        am  = batch["audio_mask"].to(device)
        vf  = batch["visual_features"].to(device)
        lbl = batch["labels"].to(device)

        a_global = audio_temp(af, mask=am)
        v_global = visual_temp(vf)
        a_proj   = audio_proj_h(a_global)
        logits   = detector(a_global, v_global, a_proj)

        loss, _ = loss_fn(logits, lbl)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_labels.extend(lbl.cpu().int().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    avg_loss = total_loss / max(1, len(val_loader))
    try:
        auc  = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc  = 0.0
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, auc, f1, bacc


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def run_detector_training() -> None:
    """Fine-tune the fusion detector on top of frozen ART representations."""
    # ── Data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv(SPLITS_DIR / "train_preprocessed.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val_preprocessed.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_preprocessed.csv")

    wav2vec2, feat_extractor, mobilenet, normalize = load_frozen_encoders(DEVICE)

    train_loader, val_loader, _ = build_dataloaders(
        train_df, val_df, test_df,
        wav2vec2, feat_extractor, mobilenet, normalize,
        batch_size=BATCH_SIZE,
    )

    # ── Models ───────────────────────────────────────────────────────────
    audio_temp  = AudioTemporalHead(AUDIO_FEAT_DIM,  256, TEMPORAL_DIM).to(DEVICE)
    visual_temp = VisualTemporalHead(VISUAL_FEAT_DIM, 256, TEMPORAL_DIM).to(DEVICE)
    audio_proj_h  = ProjectionHead(TEMPORAL_DIM, PROJ_DIM).to(DEVICE)
    visual_proj_h = ProjectionHead(TEMPORAL_DIM, PROJ_DIM).to(DEVICE)
    detector    = FusionDetectorMLP().to(DEVICE)

    # Load best ART checkpoint
    ckpt = torch.load(CKPT_DIR / f"art_epoch{ART_EPOCHS}.pt", map_location=DEVICE)
    audio_temp.load_state_dict(ckpt["audio_temp_state"])
    visual_temp.load_state_dict(ckpt["visual_temp_state"])
    audio_proj_h.load_state_dict(ckpt["audio_proj_state"])
    visual_proj_h.load_state_dict(ckpt["visual_proj_state"])
    print(f"Loaded ART checkpoint from epoch {ART_EPOCHS}")

    # Freeze ART heads
    for m in [audio_temp, visual_temp, audio_proj_h, visual_proj_h]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Loss & optimizer
    loss_fn   = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = AdamW(detector.parameters(), lr=DETECTOR_LR, weight_decay=1e-2)

    print(f"\n{'='*60}")
    print("DETECTOR FINE-TUNING")
    print(f"{'='*60}")
    print(f"Epochs  : {DETECTOR_EPOCHS}")
    print(f"LR      : {DETECTOR_LR}")
    print(f"Patience: {PATIENCE}")
    print(f"{'='*60}\n")

    best_auc     = 0.0
    patience_cnt = 0

    for epoch in range(1, DETECTOR_EPOCHS + 1):
        detector.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Det Epoch {epoch}/{DETECTOR_EPOCHS}")

        for batch in pbar:
            af  = batch["audio_features"].to(DEVICE)
            am  = batch["audio_mask"].to(DEVICE)
            vf  = batch["visual_features"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            with torch.no_grad():
                a_global = audio_temp(af, mask=am)
                v_global = visual_temp(vf)
                a_proj   = audio_proj_h(a_global)

            logits     = detector(a_global, v_global, a_proj)
            loss, _    = loss_fn(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        val_loss, auc, f1, bacc = validate(
            val_loader, audio_temp, visual_temp,
            audio_proj_h, visual_proj_h, detector, loss_fn, DEVICE,
        )
        print(
            f"  Epoch {epoch} | val_loss={val_loss:.4f} "
            f"AUC={auc:.4f} F1={f1:.4f} BalAcc={bacc:.4f}"
        )

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            patience_cnt = 0
            torch.save(detector.state_dict(), CKPT_DIR / "best_detector.pt")
            print(f"  New best AUC: {best_auc:.4f} — checkpoint saved.")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print("\nDetector training complete.")


if __name__ == "__main__":
    run_detector_training()
