"""
inference.py - Window-level inference and video-level aggregation.

Functions
---------
predict_windows
    Run inference on all windows in a DataLoader; returns per-window results.

aggregate_windows_to_video
    Aggregate window-level probabilities to a single video-level score.

evaluate_video_level
    Compute final video-level metrics and print a summary.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from .config import CKPT_DIR, DEVICE, TEMPORAL_DIM, AUDIO_FEAT_DIM, VISUAL_FEAT_DIM, PROJ_DIM
from .models.detector import FusionDetectorMLP
from .models.heads import AudioTemporalHead, VisualTemporalHead, ProjectionHead


AggMethod = Literal["mean", "max", "topk", "align_weighted"]


# ──────────────────────────────────────────────
# Window-level inference
# ──────────────────────────────────────────────

@torch.no_grad()
def predict_windows(
    dataloader,
    audio_temp,
    visual_temp,
    audio_proj_h,
    visual_proj_h,
    detector,
    device: torch.device,
) -> list[dict]:
    """
    Run inference on every window in *dataloader*.

    Returns
    -------
    list[dict]
        Each entry has keys: ``"id"``, ``"prob"``, ``"alignment"``, ``"label"``.
    """
    for m in [audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector]:
        m.eval()

    results = []

    for batch in tqdm(dataloader, desc="Inference"):
        af  = batch["audio_features"].to(device)
        am  = batch["audio_mask"].to(device)
        vf  = batch["visual_features"].to(device)
        ids = batch["id"]
        lbl = batch.get("labels", torch.zeros(len(ids)))

        a_global = audio_temp(af, mask=am)
        v_global = visual_temp(vf)
        a_proj   = audio_proj_h(a_global)
        v_proj   = visual_proj_h(v_global)

        logits      = detector(a_global, v_global, a_proj)
        probs       = torch.sigmoid(logits)
        align_scores = F.cosine_similarity(a_proj, v_proj, dim=1)

        for i in range(len(ids)):
            results.append({
                "id":        ids[i],
                "prob":      probs[i].item(),
                "alignment": align_scores[i].item(),
                "label":     int(lbl[i].item()),
            })

    return results


# ──────────────────────────────────────────────
# Video-level aggregation
# ──────────────────────────────────────────────

def aggregate_windows_to_video(
    window_results: list[dict],
    method: AggMethod = "mean",
) -> float:
    """
    Aggregate per-window probabilities to a single video-level score.

    Parameters
    ----------
    window_results : List of window dicts (must contain ``"prob"`` and ``"alignment"``).
    method         : One of ``"mean"``, ``"max"``, ``"topk"``, ``"align_weighted"``.
                     - ``"mean"`` : simple average.
                     - ``"max"``  : highest window probability.
                     - ``"topk"`` : average of top 33 % windows by probability.
                     - ``"align_weighted"`` : weight by (1 − alignment score);
                       high mis-alignment indicates manipulation.

    Returns
    -------
    float
        Video-level deepfake probability in [0, 1].
    """
    probs  = np.array([r["prob"]      for r in window_results])
    aligns = np.array([r["alignment"] for r in window_results])

    if method == "mean":
        return float(probs.mean())
    elif method == "max":
        return float(probs.max())
    elif method == "topk":
        k = max(1, int(len(probs) * 0.33))
        top_idx = np.argsort(probs)[-k:]
        return float(probs[top_idx].mean())
    elif method == "align_weighted":
        weights = 1.0 - aligns
        weights = weights / (weights.sum() + 1e-8)
        return float((probs * weights).sum())
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")


# ──────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────

def evaluate_video_level(
    win_df: pd.DataFrame,
    threshold: float = 0.5,
    method: AggMethod = "mean",
) -> None:
    """
    Aggregate window predictions to video level and print metrics.

    Parameters
    ----------
    win_df    : DataFrame with columns ``id``, ``prob``, ``alignment``, ``label``.
                ``id`` should be a video identifier (windows share the same prefix).
    threshold : Classification threshold for binary prediction.
    method    : Aggregation method passed to :func:`aggregate_windows_to_video`.
    """
    # Group by video id prefix (split off window index)
    win_df = win_df.copy()
    win_df["video_id"] = win_df["id"].apply(lambda x: str(x).rsplit("_w", 1)[0])

    video_results = []
    for vid, grp in win_df.groupby("video_id"):
        rows   = grp.to_dict("records")
        score  = aggregate_windows_to_video(rows, method=method)
        label  = int(grp["label"].mode()[0])
        video_results.append({"video_id": vid, "score": score, "label": label})

    res_df = pd.DataFrame(video_results)
    preds  = (res_df["score"] >= threshold).astype(int)
    labels = res_df["label"]

    try:
        auc = roc_auc_score(labels, res_df["score"])
    except Exception:
        auc = float("nan")

    f1   = f1_score(labels, preds, zero_division=0)
    bacc = balanced_accuracy_score(labels, preds)

    print("\n" + "=" * 50)
    print("VIDEO-LEVEL EVALUATION")
    print("=" * 50)
    print(f"  Aggregation : {method}")
    print(f"  Threshold   : {threshold}")
    print(f"  Videos      : {len(res_df)}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  Balanced Acc: {bacc:.4f}")
    print("=" * 50)

    return res_df


def load_models_for_inference(device: torch.device):
    """
    Instantiate and load the best saved checkpoints for all model components.

    Returns
    -------
    tuple
        ``(audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector)``
        all moved to *device* and set to eval mode.
    """
    audio_temp    = AudioTemporalHead(AUDIO_FEAT_DIM,  256, TEMPORAL_DIM).to(device)
    visual_temp   = VisualTemporalHead(VISUAL_FEAT_DIM, 256, TEMPORAL_DIM).to(device)
    audio_proj_h  = ProjectionHead(TEMPORAL_DIM, 128).to(device)
    visual_proj_h = ProjectionHead(TEMPORAL_DIM, 128).to(device)
    detector      = FusionDetectorMLP().to(device)

    # Load ART checkpoint
    from .config import ART_EPOCHS
    ckpt = torch.load(CKPT_DIR / f"art_epoch{ART_EPOCHS}.pt", map_location=device)
    audio_temp.load_state_dict(ckpt["audio_temp_state"])
    visual_temp.load_state_dict(ckpt["visual_temp_state"])
    audio_proj_h.load_state_dict(ckpt["audio_proj_state"])
    visual_proj_h.load_state_dict(ckpt["visual_proj_state"])

    # Load best detector
    detector.load_state_dict(
        torch.load(CKPT_DIR / "best_detector.pt", map_location=device)
    )

    for m in [audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector]:
        m.eval()

    return audio_temp, visual_temp, audio_proj_h, visual_proj_h, detector
