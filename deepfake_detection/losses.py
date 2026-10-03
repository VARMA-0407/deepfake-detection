"""
losses.py - Loss functions for ART contrastive pre-training.

Functions
---------
info_nce_loss
    Symmetric InfoNCE contrastive loss used during ART pre-training.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    audio_proj:  torch.Tensor,
    visual_proj: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss.

    Within a batch of *B* aligned audio-visual pairs,
    diagonal elements are treated as positives and all
    off-diagonal pairs as negatives.

    Parameters
    ----------
    audio_proj  : ``(B, D)`` L2-normalised audio projections.
    visual_proj : ``(B, D)`` L2-normalised visual projections.
    temperature : Scaling factor for logit sharpness.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    B = audio_proj.shape[0]
    sim_matrix = (audio_proj @ visual_proj.T) / temperature   # (B, B)
    labels     = torch.arange(B, device=audio_proj.device)

    loss_a2v = F.cross_entropy(sim_matrix,   labels)
    loss_v2a = F.cross_entropy(sim_matrix.T, labels)

    return 0.5 * (loss_a2v + loss_v2a)
