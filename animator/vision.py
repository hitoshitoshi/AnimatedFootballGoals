import cv2
import numpy as np
from typing import Optional, Tuple

def combine_lines_geom(lines: np.ndarray,
                       dist_thresh: float = 100,
                       angle_eps_rad: float = np.deg2rad(5)) -> list[tuple[int,int,int,int]]:
    """
    Merge line segments if they share alignment and lie close to the same infinite line.

    Args:
        lines: Array of segments [[x1,y1,x2,y2], ...]
        dist_thresh: Max distance of endpoints to merge
        angle_eps_rad: Max angle difference for grouping
    Returns:
        List of merged line endpoints [(x1,y1,x2,y2), ...]
    """
    if lines.ndim == 3:
        lines = lines[:, 0]

    groups: list[list[np.ndarray]] = []
    for seg in lines:
        x1, y1, x2, y2 = seg
        vec = np.array([x2-x1, y2-y1], float)
        length = np.hypot(*vec)
        if length == 0:
            continue
        n = vec / length
        placed = False
        for g in groups:
            gx1, gy1, gx2, gy2 = g[0]
            gvec = np.array([gx2-gx1, gy2-gy1], float)
            gvec /= np.hypot(*gvec)
            # Check angle similarity
            if abs(np.arccos(np.clip(np.dot(n, gvec), -1, 1))) > angle_eps_rad:
                continue
            # Distance of segment endpoints to group line
            a, b = -gvec[1], gvec[0]
            c = -(a*gx1 + b*gy1)
            dist1 = abs(a*x1 + b*y1 + c)
            dist2 = abs(a*x2 + b*y2 + c)
            if max(dist1, dist2) > dist_thresh:
                continue
            g.append(seg)
            placed = True
            break
        if not placed:
            groups.append([seg])

    merged = []
    for g in groups:
        pts = np.vstack(g).reshape(-1, 2)
        pts_mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - pts_mean)
        direction = vt[0]
        proj = (pts - pts_mean) @ direction
        p1 = pts_mean + proj.min() * direction
        p2 = pts_mean + proj.max() * direction
        merged.append(tuple(map(int, (*p1, *p2))))
    return merged


def hough_lines(mask: np.ndarray, params: dict) -> Optional[np.ndarray]:
    """Run Probabilistic Hough Transform on a binary mask."""
    return cv2.HoughLinesP(mask, **params)


def longest_valid_line(lines: Optional[np.ndarray]) -> Optional[Tuple[Tuple[int,int],Tuple[int,int]]]:
    """
    From detected segments, select the longest non-horizontal/vertical line.

    Args:
        lines: Array of [[x1,y1,x2,y2], ...]
    Returns:
        Endpoint pair ((x1,y1), (x2,y2)) or None if no valid line.
    """
    if lines is None or len(lines) == 0:
        return None
    best, best_len = None, -1
    for x1,y1,x2,y2 in lines[:,0]:
        dx, dy = x2-x1, y2-y1
        ang = np.degrees(np.arctan2(dy,dx)) % 180
        # ignore near-horizontal/vertical lines
        if 10 < ang < 170:
            length = dx*dx + dy*dy
            if length > best_len:
                best, best_len = ((x1,y1),(x2,y2)), length
    return best
