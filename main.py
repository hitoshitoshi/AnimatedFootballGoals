from __future__ import annotations
import cv2, numpy as np, sys
from typing import Optional, Tuple, List
from pathlib import Path
from ultralytics import YOLO

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer‑goal animator (cartoon pitch with posterised real players).

Features:
- Detects and draws pitch lines and goal geometry using HSV color filtering and Hough transforms.
- Tracks the ball via YOLO and color-based centroid tracking, filling its contour with the tracked average color and a black outline.
- Segments players near the ball and renders posterised cut‑outs using k-means clustering for dominant colors.

"""

class CONFIG:
    """Configuration parameters for video processing and detection."""
    # Output video settings
    OUTPUT_NAME = "output.mp4"
    TARGET_FPS  = 10                     # Output and processing frames per second

    # YOLO ball detection
    BALL_MODEL  = "yolo11x.pt"        # YOLO model for ball detection
    BALL_LABEL  = "sports ball"       # Class label for ball
    BALL_CONF   = 0.5                  # Confidence threshold for ball detection

    # YOLO segmentation model for players
    SEG_MODEL   = "yolo11x-seg.pt"    # YOLO model for player segmentation
    SEG_CONF    = 0.2                  # Confidence threshold for segmentation
    SEG_IOU     = 0.4                  # IoU threshold for NMS
    SEARCH_SIZE = 1200                 # ROI size for segmentation (square)

    # Drawing parameters scaled to 2160p reference
    BASELINE_H  = 2160                 # Reference height for scaling
    DRAW_DIST   = 400                  # Max radius from ball to segment players

    # Color tracker parameters
    COLOUR_THRESH = 25.0               # Color difference threshold for mask
    MOMENTUM_K    = 3                  # Exponential smoothing factor

    # HSV bands for line detection
    KERNEL       = (3, 3)              # Kernel size for morphological ops
    HSV_BAND_1   = (np.array([20, 20, 140]),  np.array([100, 100, 255]))  # Pitch lines
    HSV_BAND_2   = (np.array([75, 0, 150]),   np.array([180, 10, 255]))    # Crossbar and posts

    # Hough transform parameters for thick and thin lines
    HOUGH_1 = dict(
        rho=1.0,
        theta=np.pi/180,
        threshold=700,
        minLineLength=300,
        maxLineGap=100
    )
    HOUGH_2 = dict(
        rho=1.0,
        theta=np.pi/180,
        threshold=100,
        minLineLength=500,
        maxLineGap=100
    )

# Drawing style constants
PITCH_COLOR = (20, 100, 20)  # Green pitch background
LINE_COLOR  = (255, 255, 255)  # White for lines and goal
GOAL_COLOR  = LINE_COLOR
LINE_W      = 15                # Stroke width for lines
KERNEL_STRUCT = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG.KERNEL)


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


def posterise_player(raw: np.ndarray, mask: np.ndarray, k: int = 4):
    """
    Extract and posterise a player region using k-means on Lab colors.

    Args:
        raw: original frame
        mask: binary mask of player region
        k: number of clusters/colors
    Returns:
        (RGB sprite, alpha mask)
    """
    x,y,w,h = cv2.boundingRect(mask)
    if w<4 or h<10:
        return np.zeros((h,w,3),np.uint8), np.zeros((h,w),np.uint8)
    crop_rgb  = raw[y:y+h,x:x+w].copy()
    crop_mask = mask[y:y+h,x:x+w]
    crop_rgb[crop_mask==0] = 0
    blur = cv2.bilateralFilter(crop_rgb,5,50,50)
    lab = cv2.cvtColor(blur,cv2.COLOR_BGR2Lab).reshape(-1,3)
    nz = np.where(crop_mask.flatten()>0)[0]
    if len(nz)<k:
        return crop_rgb, crop_mask
    samples = lab[nz].astype(np.float32)
    _,labels,centers = cv2.kmeans(
        samples, k, None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),
        1, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)
    lab[nz] = centers[labels.flatten()]
    sprite = cv2.cvtColor(lab.reshape(h,w,3),cv2.COLOR_Lab2BGR)
    return sprite, crop_mask


def paste_sprite(canvas: np.ndarray, sprite: np.ndarray,
                 alpha: np.ndarray, topleft: Tuple[int,int]) -> None:
    """
    Alpha-blend a sprite onto the canvas at the given top-left position.

    Args:
        canvas: destination image
        sprite: BGR sprite image
        alpha: mask (0-255)
        topleft: (x,y) coordinate
    """
    x,y = topleft
    h,w = alpha.shape
    roi = canvas[y:y+h, x:x+w]
    maskf = alpha.astype(float)/255.0
    for c in range(3):
        roi[...,c] = (1-maskf)*roi[...,c] + maskf*sprite[...,c]


def stylise(frame: np.ndarray, width: int, height: int,
            centre: Optional[Tuple[int,int]], radius: Optional[int],
            avg_col: Optional[np.ndarray], seg_model: YOLO, cfg,
            show_ball: bool) -> np.ndarray:
    """
    Render a single annotated frame with pitch, goal, ball, and players.

    Args:
        frame: original BGR frame
        width,height: frame dimensions
        centre: ball center or None
        radius: ball radius or None
        avg_col: tracked BGR color or None
        seg_model: pretrained YOLO segmenter
        cfg: CONFIG containing thresholds
        show_ball: whether to draw ball and process players

    Returns:
        Annotated image with pitch background and overlays.
    """
    drawings: list[tuple[str,object]] = []

    # Queue ball for drawing if available
    if show_ball and centre and radius and avg_col is not None:
        drawings.append(("ball", (centre, radius, avg_col)))

    # Detect pitch lines via HSV mask + Hough
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_pitch = cv2.inRange(hsv, *cfg.HSV_BAND_1)
    mask_pitch = cv2.morphologyEx(mask_pitch, cv2.MORPH_CLOSE, KERNEL_STRUCT, 2)
    lp = hough_lines(mask_pitch, cfg.HOUGH_1)
    if lp is not None:
        merged = combine_lines_geom(lp)
        drawings.append(("pitch_lines", merged))

    # Detect crossbar lines similarly
    mask_cross = cv2.inRange(hsv, *cfg.HSV_BAND_2)
    mask_cross = cv2.morphologyEx(mask_cross, cv2.MORPH_CLOSE, KERNEL_STRUCT, 2)
    lc = hough_lines(cv2.Canny(mask_cross,50,100), cfg.HOUGH_2)
    if lc is not None:
        drawings.append(("cross_lines", lc))

    # Segment players around the ball if ball detected
    if centre:
        scale_h = min(width,height)/cfg.BASELINE_H
        # Adjust Hough thresholds by resolution
        hough1 = {**cfg.HOUGH_1,
                  "threshold": int(cfg.HOUGH_1["threshold"]*scale_h),
                  "minLineLength": int(cfg.HOUGH_1["minLineLength"]*scale_h),
                  "maxLineGap": int(cfg.HOUGH_1["maxLineGap"]*scale_h)}
        lp = hough_lines(mask_pitch, hough1)

        hough2 = {**cfg.HOUGH_2,
                  "threshold": int(cfg.HOUGH_2["threshold"]*scale_h),
                  "minLineLength": int(cfg.HOUGH_2["minLineLength"]*scale_h),
                  "maxLineGap": int(cfg.HOUGH_2["maxLineGap"]*scale_h)}
        lc = hough_lines(mask_cross, hough2)

        # Define ROI around ball for segmentation
        scaled_search = int(cfg.SEARCH_SIZE*scale_h)
        half = scaled_search//2
        cx,cy = centre
        x1,y1 = max(cx-half,0), max(cy-half,0)
        x2,y2 = min(cx+half,width), min(cy+half,height)
        crop = frame[y1:y2, x1:x2]

        # Run segmentation on ROI
        seg = seg_model(crop, task="segment",
                        conf=cfg.SEG_CONF, iou=cfg.SEG_IOU, verbose=False)[0]
        if seg.masks is not None:
            masks, classes = seg.masks.data.cpu().numpy(), seg.boxes.cls.cpu().numpy()
            draw_r = int(cfg.DRAW_DIST*scale_h)
            for m,cls in zip(masks, classes):
                if int(cls) != 0:  # only sports ball class has cls=0
                    continue
                mb = (m*255).astype(np.uint8)
                ch, cw = y2-y1, x2-x1
                if mb.shape != (ch,cw):
                    mb = cv2.resize(mb,(cw,ch),cv2.INTER_NEAREST)
                bx,by,bw,bh = cv2.boundingRect(mb)
                pcx, pcy = x1+bx+bw//2, y1+by+bh//2
                # Skip players far from ball
                if np.hypot(pcx-cx,pcy-cy) > draw_r:
                    continue
                full = np.zeros(frame.shape[:2],np.uint8)
                full[y1:y2,x1:x2] = mb
                sprite, alpha = posterise_player(frame, full, 4)
                drawings.append(("player", (sprite, alpha, (x1+bx,y1+by))))

    # Create base pitch image
    out = np.full_like(frame, PITCH_COLOR)

    # Draw pitch lines
    for kind, data in drawings:
        if kind == "pitch_lines":
            for x1,y1,x2,y2 in data:
                cv2.line(out, (x1,y1), (x2,y2), GOAL_COLOR, LINE_W)

    # Draw goal crossbar and posts
    crs = [d for k,d in drawings if k=="cross_lines"]
    ln = longest_valid_line(np.vstack(crs) if crs else None)
    if ln:
        (x1c,y1c),(x2c,y2c) = ln
        # crossbar
        cv2.line(out,(x1c,y1c),(x2c,y2c),GOAL_COLOR,LINE_W)
        # left post
        cv2.line(out,(x1c,y1c),(x1c,y1c+375),GOAL_COLOR,LINE_W)
        # right post
        cv2.line(out,(x2c,y2c),(x2c,y2c+425),GOAL_COLOR,LINE_W)

    # Draw ball on top
    for kind, data in drawings:
        if kind == "ball":
            (cc, rr, col) = data
            cv2.circle(out, cc, rr, col.tolist(), -1)
            cv2.circle(out, cc, rr, (0,0,0), 2)

    # Overlay player sprites
    for kind, data in drawings:
        if kind == "player":
            sprite, alpha, topleft = data
            paste_sprite(out, sprite, alpha, topleft)

    return out


def annotate_video(src: str | Path, cfg=CONFIG) -> None:
    """
    Process input video, detect and track ball, stylise frames, and save output.

    Args:
        src: Path to input video file
        cfg: Configuration class instance
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {src}")

    # Get properties and compute frame skip interval
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, round(in_fps / cfg.TARGET_FPS))

    # Initialize video writer
    out = cv2.VideoWriter(
        cfg.OUTPUT_NAME,
        cv2.VideoWriter_fourcc(*'avc1'),
        cfg.TARGET_FPS,
        (width, height)
    )

    # Load and warm up YOLO models
    ball_yolo = YOLO(cfg.BALL_MODEL)
    seg_yolo = YOLO(cfg.SEG_MODEL)
    dummy = np.zeros((cfg.SEARCH_SIZE, cfg.SEARCH_SIZE, 3), np.uint8)
    ball_yolo(dummy, conf=0.01, verbose=False)
    seg_yolo(dummy, task="segment", conf=0.01, iou=0.1, verbose=False)

    buffer: list[np.ndarray] = []  # store frames until first detection
    centre, radius, avg_col = None, None, None
    frame_idx = -1
    yolo_half = 800 // 2  # half-size of YOLO ROI

    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % interval != 0:
            continue

        # Buffer frames while waiting for first ball detection
        if centre is None:
            buffer.append(fr.copy())

        # Perform YOLO detection with ROI if available
        if centre:
            cx, cy = centre
            x1_roi = max(cx-yolo_half,0)
            y1_roi = max(cy-yolo_half,0)
            x2_roi = min(cx+yolo_half,width)
            y2_roi = min(cy+yolo_half,height)
            roi = fr[y1_roi:y2_roi, x1_roi:x2_roi]
            res = ball_yolo(roi, conf=cfg.BALL_CONF, verbose=False)[0]
        else:
            x1_roi, y1_roi = 0,0
            res = ball_yolo(fr, conf=cfg.BALL_CONF, verbose=False)[0]

        # Collect ball candidates from YOLO
        candidates: list[tuple[Tuple[int,int],int]] = []
        for box in res.boxes:
            cls = int(box.cls[0])
            if ball_yolo.names[cls] != cfg.BALL_LABEL:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            fx1, fy1 = bx1+x1_roi, by1+y1_roi
            fx2, fy2 = bx2+x1_roi, by2+y1_roi
            cdet = ((fx1+fx2)//2, (fy1+fy2)//2)
            rdet = ((fx2-fx1)+(fy2-fy1))//4
            candidates.append((cdet, rdet))

        # Perform color tracking mask
        if centre:
            msk, new_c = vectorised_colour_search(
                fr, centre, (width//10, height//10), avg_col, cfg.COLOUR_THRESH)
        else:
            msk, new_c = np.zeros(fr.shape[:2], np.uint8), None

        # Choose best detection if multiple
        det = None
        if candidates:
            if centre and new_c is not None and len(candidates)>1:
                dists = [np.hypot(c[0][0]-new_c[0], c[0][1]-new_c[1]) for c in candidates]
                det = candidates[int(np.argmin(dists))]
            else:
                det = candidates[0]

        # First detection: initialize tracker and backfill buffered frames
        if det and centre is None:
            centre, radius = det
            m_init = np.zeros(fr.shape[:2], np.uint8)
            cv2.circle(m_init, centre, radius, 255, -1)
            avg_col = np.array(cv2.mean(fr, mask=m_init)[:3])
            for prev in reversed(buffer):
                mask_prev, c_prev = vectorised_colour_search(
                    prev, centre, (width//10, height//10), avg_col, cfg.COLOUR_THRESH)
                show_ball = False
                if mask_prev.any():
                    new_avg = np.array(cv2.mean(prev, mask=mask_prev)[:3])
                    avg_col = update_running_colour(avg_col, new_avg, mask_prev.sum()//255, (width//10)*(height//10))
                    centre = tuple(c_prev)
                    show_ball = True
                out_prev = stylise(prev, width, height, centre, radius, avg_col, seg_yolo, cfg, show_ball)
                out.write(out_prev)
            buffer.clear()

        # Update tracker each frame: decide between YOLO or colour
        show_ball = False
        if det and centre and new_c is not None and msk.any():
            dist = np.hypot(det[0][0]-new_c[0], det[0][1]-new_c[1])
            if dist>100:
                centre = det[0]
                show_ball = True
            else:
                new_avg = np.array(cv2.mean(fr, mask=msk)[:3])
                avg_col = update_running_colour(avg_col, new_avg, msk.sum()//255, (width//10)*(height//10))
                centre = tuple(new_c)
                show_ball = True
        elif msk.any() and centre:
            new_avg = np.array(cv2.mean(fr, mask=msk)[:3])
            avg_col = update_running_colour(avg_col, new_avg, msk.sum()//255, (width//10)*(height//10))
            centre = tuple(new_c)
            show_ball = True

        # Stylise and write current frame
        out_f = stylise(fr, width, height, centre, radius, avg_col, seg_yolo, cfg, show_ball)
        out.write(out_f)

    cap.release()
    out.release()
    print(f"Finished → {cfg.OUTPUT_NAME}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv)>1 else "TestGoal2.mp4"
    annotate_video(src)
