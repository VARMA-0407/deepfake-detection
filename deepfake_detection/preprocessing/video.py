"""
video.py - Video / lip-ROI preprocessing utilities.

Functions:
    resample_indices        : Linearly resample frame indices to a target count.
    smooth_boxes            : Temporally smooth bounding boxes with a moving average.
    extract_lip_roi_window  : Extract & save a lip-ROI video segment as a .npz file.
    load_video_roi_tensor   : Load a saved .npz ROI tensor back into memory.
"""

from pathlib import Path

import cv2
import numpy as np
import torch


def resample_indices(orig_len: int, target_len: int) -> np.ndarray:
    """Return *target_len* evenly-spaced integer indices into a sequence of *orig_len*."""
    if orig_len == target_len:
        return np.arange(target_len)
    indices = np.linspace(0, orig_len - 1, target_len)
    return np.round(indices).astype(int)


def smooth_boxes(boxes: list, window: int = 5) -> list:
    """
    Temporally smooth a list of [x, y, w, h] bounding boxes
    using a uniform moving-average kernel of width *window*.
    """
    if len(boxes) < window:
        return boxes
    boxes_arr = np.array(boxes, dtype=float)
    kernel = np.ones(window) / window
    for i in range(4):
        boxes_arr[:, i] = np.convolve(boxes_arr[:, i], kernel, mode="same")
    return boxes_arr.astype(int).tolist()


def extract_lip_roi_window(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_npz: Path,
    crop_size: int = 112,
    padding: float = 0.35,
    smooth_win: int = 5,
    target_fps: int = 25,
) -> bool:
    """
    Extract the lip-region of interest (ROI) from a video window and save it.

    Uses MediaPipe face detection to locate faces, then crops the lower 40 %
    of the bounding box (the lip area) with *padding* added on all sides.

    Parameters
    ----------
    video_path  : Path to input video file.
    start_sec   : Window start time (seconds).
    end_sec     : Window end time (seconds).
    out_npz     : Destination .npz file; stores key ``"frames"``.
    crop_size   : Output spatial resolution (pixels, square).
    padding     : Relative padding added around the lip region.
    smooth_win  : Moving-average window for temporal box smoothing.
    target_fps  : Number of output frames per second.

    Returns
    -------
    bool
        True on success, False otherwise.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return False

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return False

    # Temporal resampling
    indices = resample_indices(len(frames), target_fps)
    frames  = [frames[i] for i in indices]

    # Face detection
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    boxes = []
    for frame in frames:
        results = detector.process(frame)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x     = int(bbox.xmin  * w)
            y     = int(bbox.ymin  * h)
            box_w = int(bbox.width  * w)
            box_h = int(bbox.height * h)
            boxes.append([x, y, box_w, box_h])
        else:
            if boxes:
                boxes.append(boxes[-1])
            else:
                h, w = frame.shape[:2]
                boxes.append([0, 0, w, h])
    detector.close()

    boxes = smooth_boxes(boxes, window=smooth_win)

    crops = []
    for frame, (x, y, w, h) in zip(frames, boxes):
        lip_y  = y + int(h * 0.6)
        lip_h  = int(h * 0.4)
        pad_h  = int(lip_h * padding)
        pad_w  = int(w     * padding)
        y1 = max(0, lip_y - pad_h)
        y2 = min(frame.shape[0], lip_y + lip_h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(frame.shape[1], x + w + pad_w)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (crop_size, crop_size))
        crops.append(crop)

    # (T, 3, H, W) float in [0, 1]
    crops_tensor = (
        torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    )
    np.savez_compressed(out_npz, frames=crops_tensor.numpy())
    return True


def load_video_roi_tensor(npz_path: Path) -> torch.Tensor:
    """
    Load a lip-ROI tensor previously saved by :func:`extract_lip_roi_window`.

    Returns
    -------
    torch.Tensor
        Shape ``(T, 3, H, W)`` float32 in [0, 1].
    """
    data = np.load(npz_path)
    return torch.from_numpy(data["frames"])
