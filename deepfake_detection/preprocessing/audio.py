"""
audio.py - Audio preprocessing utilities.

Functions:
    slice_audio_ffmpeg  : Extract an audio clip from a video file using FFmpeg.
    load_audio_tensor   : Load a WAV file as a fixed-length 1-D torch tensor.
"""

import subprocess
import torch
import torch.nn.functional as F
from pathlib import Path


def slice_audio_ffmpeg(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_wav: Path,
    sr: int = 16_000,
) -> bool:
    """
    Extract a mono audio slice from *video_path* in [start_sec, end_sec)
    and write it to *out_wav* at sample-rate *sr*.

    Returns
    -------
    bool
        True on success, False if FFmpeg failed or the output is empty.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL, timeout=10)
        return out_wav.exists() and out_wav.stat().st_size > 0
    except Exception:
        return False


def load_audio_tensor(
    audio_path: Path,
    target_sr: int = 16_000,
    target_len: int = 16_000,
) -> torch.Tensor:
    """
    Load *audio_path* as a 1-D float tensor of exactly *target_len* samples.

    Resamples if the on-disk sample rate differs from *target_sr*.
    Pads with zeros if shorter than *target_len*; truncates if longer.

    Returns
    -------
    torch.Tensor
        Shape ``(target_len,)``
    """
    import torchaudio

    try:
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)  # stereo -> mono

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] < target_len:
            pad_len = target_len - waveform.shape[0]
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:target_len]

        return waveform
    except Exception:
        return torch.zeros(target_len)
