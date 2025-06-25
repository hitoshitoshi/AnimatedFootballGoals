import numpy as np
from typing import Tuple

from .config import CONFIG

def vectorised_colour_search(frame: np.ndarray, centre: Tuple[int,int],
                             window: Tuple[int,int], avg_colour: np.ndarray,
                             thresh: float) -> Tuple[np.ndarray,np.ndarray]:
    """
    Build a mask of pixels within color threshold around avg_colour in a search window.

    Args:
        frame: BGR image
        centre: previous ball centre (x,y)
        window: half-width and half-height of search window
        avg_colour: reference BGR color
        thresh: distance threshold
    Returns:
        (full-frame mask, new centroid)
    """
    h, w = frame.shape[:2]
    win_w, win_h = window
    cx, cy = centre
    x1, y1 = max(cx-win_w,0), max(cy-win_h,0)
    x2, y2 = min(cx+win_w,w), min(cy+win_h,h)

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    diff = np.linalg.norm(roi - avg_colour, axis=2)
    mask_local = (diff < thresh).astype(np.uint8)

    if mask_local.sum() == 0:
        return np.zeros((h,w),np.uint8), np.array(centre)

    mask_full = np.zeros((h,w),np.uint8)
    mask_full[y1:y2, x1:x2] = mask_local * 255
    ys, xs = np.nonzero(mask_local)
    centroid = np.array([xs.mean()+x1, ys.mean()+y1], dtype=int)
    return mask_full, centroid


def update_running_colour(avg: np.ndarray, new_avg: np.ndarray,
                          mask_area: int, window_area: int,
                          k: int = CONFIG.MOMENTUM_K) -> np.ndarray:
    """
    Smoothly update the tracked ball color using exponential momentum.

    mask_area: area of detected mask
    window_area: total search window area
    k: exponent for momentum
    """
    ratio = (mask_area/window_area) ** k
    return (1-ratio)*avg + ratio*new_avg
