"""
detector.py - Fusion MLP detector and Focal Loss.

Classes
-------
FocalLoss
    Focal Loss for binary classification on imbalanced deepfake datasets.
    alpha controls positive-class weight; gamma focuses on hard examples.

FusionDetectorMLP
    Three-layer MLP that fuses audio, visual, and audio-projection embeddings
    to produce a single real/fake logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class (fake).
    gamma : float
        Focusing exponent; higher values down-weight easy examples more.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        logits : ``(B,)`` unnormalised predictions.
        labels : ``(B,)`` binary ground-truth (0 = real, 1 = fake).

        Returns
        -------
        loss   : scalar focal loss.
        info   : dict with ``"bce"`` and ``"focal_weight"`` diagnostics.
        """
        probs        = torch.sigmoid(logits)
        pt           = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        bce          = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        alpha_weight = torch.where(
            labels == 1,
            torch.full_like(labels, self.alpha, dtype=torch.float),
            torch.full_like(labels, 1 - self.alpha, dtype=torch.float),
        )
        loss = (alpha_weight * focal_weight * bce).mean()
        return loss, {"bce": bce.mean().item(), "focal_weight": focal_weight.mean().item()}


class FusionDetectorMLP(nn.Module):
    """
    Fusion MLP for deepfake detection.

    Concatenates audio global (256), visual global (256), and audio projection (128)
    embeddings → 640-d input → hidden layers → single logit.

    Parameters
    ----------
    audio_dim   : Dimension of audio temporal embedding (default 256).
    visual_dim  : Dimension of visual temporal embedding (default 256).
    proj_dim    : Dimension of audio projection embedding (default 128).
    hidden_dim  : Hidden layer width (default 256).
    dropout     : Dropout probability (default 0.3).
    """

    def __init__(
        self,
        audio_dim:  int   = 256,
        visual_dim: int   = 256,
        proj_dim:   int   = 128,
        hidden_dim: int   = 256,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = audio_dim + visual_dim + proj_dim  # 640
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        audio_global:  torch.Tensor,
        visual_global: torch.Tensor,
        audio_proj:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_global  : ``(B, 256)``
        visual_global : ``(B, 256)``
        audio_proj    : ``(B, 128)`` L2-normalised

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` — logits.
        """
        fused = torch.cat([audio_global, visual_global, audio_proj], dim=1)
        return self.net(fused).squeeze(1)
