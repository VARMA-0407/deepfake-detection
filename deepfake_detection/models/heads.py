"""
heads.py - Temporal aggregation and projection heads.

Classes
-------
AudioTemporalHead
    Bidirectional LSTM: encodes variable-length Wav2Vec2 features
    ``(B, L, 768)`` → ``(B, 256)`` global embedding.

VisualTemporalHead
    Three-layer Temporal Convolutional Network (TCN): encodes
    fixed-length per-frame MobileNetV3 features
    ``(B, T, 576)`` → ``(B, 256)`` global embedding.

ProjectionHead
    Two-layer MLP: projects a 256-d embedding to a 128-d
    L2-normalised vector for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTemporalHead(nn.Module):
    """
    Bidirectional LSTM for audio temporal aggregation.

    Parameters
    ----------
    input_dim  : Wav2Vec2 hidden size (default 768).
    hidden_dim : LSTM hidden size per direction (default 256).
    output_dim : Output embedding dimension (default 256).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : ``(B, L, 768)`` audio feature sequence.
        mask : ``(B, L)`` binary mask (1 = valid, 0 = padding).

        Returns
        -------
        torch.Tensor
            Shape ``(B, 256)``.
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        # Masked mean pooling
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            lstm_out = lstm_out * mask_exp
            pooled   = lstm_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = lstm_out.mean(dim=1)

        return self.ln(self.fc(pooled))


class VisualTemporalHead(nn.Module):
    """
    Three-layer TCN for visual temporal aggregation.

    Parameters
    ----------
    input_dim  : MobileNetV3-Small feature size (default 576).
    hidden_dim : Internal channel count (default 256).
    output_dim : Output embedding dimension (default 256).
    """

    def __init__(
        self,
        input_dim: int = 576,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim,  hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(output_dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, T, C)`` visual feature sequence.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 256)``.
        """
        x = x.transpose(1, 2)                   # (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.pool(x).squeeze(-1)          # (B, 256)


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head for contrastive learning.

    Projects a *input_dim*-d embedding to a L2-normalised *output_dim*-d vector.
    """

    def __init__(self, input_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.fc1  = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : ``(B, input_dim)``

        Returns
        -------
        torch.Tensor
            Shape ``(B, output_dim)``, L2-normalised.
        """
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)
