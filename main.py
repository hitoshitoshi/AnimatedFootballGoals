#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer‑goal animator (cartoon pitch with posterised real players).

Features:
- Detects and draws pitch lines and goal geometry using HSV color filtering and Hough transforms.
- Tracks the ball via YOLO and color-based centroid tracking, filling its contour with the tracked average color and a black outline.
- Segments players near the ball and renders posterised cut‑outs using k-means clustering for dominant colors.

"""

from __future__ import annotations
import cv2, numpy as np, sys
from typing import Tuple
from pathlib import Path
from ultralytics import YOLO

from animator.config import CONFIG
from animator.tracking import vectorised_colour_search, update_running_colour
from animator.stylization import stylise

KERNEL_STRUCT = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG.KERNEL)

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
    src = sys.argv[1] if len(sys.argv)>1 else "./data/TestGoal.mp4"
    annotate_video(src)
