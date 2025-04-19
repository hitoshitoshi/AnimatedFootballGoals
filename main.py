from __future__ import annotations
import cv2, numpy as np, sys
from typing import Optional, Tuple, List
from pathlib import Path
from ultralytics import YOLO

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soccer‑goal animator (cartoon pitch, *posterised real players*).

• HSV_BAND_1  → pitch_lines  (drawn)
• HSV_BAND_2  → cross_lines  (goal geometry)
• Players rendered as posterised cut‑outs (using individual 4 dominant colors per player).

Goal posts:
After detecting the crossbar, two vertical posts extend 200 pixels downward from its endpoints.

Ball drawing enhancement:
The ball is now drawn as a circle filled with its real average color (tracked during detection)
with a small black outline.
  
Author: ChatGPT (o3)
"""


"""
Things to Do: 3h
MVP:
Make lines better 1h
Try to make it adaptable to lower quality videos 2h
"""
# ───────────────────────── CONFIG ───────────────────────── #

class CONFIG:
    # IO
    OUTPUT_NAME = "output.mp4"
    TARGET_FPS  = 10                     # final & processing fps

    # Ball detector
    BALL_MODEL  = "yolov8x.pt"
    BALL_LABEL  = "sports ball"
    BALL_CONF   = 0.5

    # Segmentation
    SEG_MODEL   = "yolo11x-seg.pt"
    SEG_CONF    = 0.2
    SEG_IOU     = 0.4
    SEARCH_SIZE = 1200                   # always 1 200 × 1 200

    # Draw‑radius (400 px at 2160 p, scaled otherwise)
    BASELINE_H  = 2160
    DRAW_DIST   = 400
    

    # Colour tracker
    COLOUR_THRESH = 25.0
    MOMENTUM_K    = 3

    # Pitch lines & goal
    KERNEL       = (3, 3)
    HSV_BAND_1   = (np.array([20, 20, 140]),  np.array([100, 100, 255]))
    HSV_BAND_2   = (np.array([75, 0, 150]),   np.array([180, 10, 255]))
    HOUGH_1      = dict(rho=1.0, theta=np.pi/180, threshold=700,
                        minLineLength=500, maxLineGap=100)
    HOUGH_2      = dict(rho=1.0, theta=np.pi/180, threshold=100,
                        minLineLength=500, maxLineGap=100)

# ───────────────────────── STYLE CONSTANTS ───────────────────────── #

PITCH_COLOR = (20, 100, 20)
LINE_COLOR  = (255, 255, 255)
GOAL_COLOR  = LINE_COLOR
LINE_W      = 6
KERNEL_STRUCT = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG.KERNEL)

# ───────────────────────── HELPER FUNCTIONS ───────────────────────── #

def hough_lines(mask: np.ndarray, params: dict) -> Optional[np.ndarray]:
    return cv2.HoughLinesP(mask, **params)

def longest_valid_line(lines: Optional[np.ndarray]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if lines is None or len(lines) == 0:
        return None
    best, best_len = None, -1
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx)) % 180
        if 10 < ang < 170:
            length = dx*dx + dy*dy
            if length > best_len:
                best, best_len = ((x1, y1), (x2, y2)), length
    return best

def vectorised_colour_search(frame: np.ndarray, centre: Tuple[int, int],
                             window: Tuple[int, int], avg_colour: np.ndarray,
                             thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    win_w, win_h = window
    cx, cy = centre
    x1, y1 = max(cx - win_w, 0), max(cy - win_h, 0)
    x2, y2 = min(cx + win_w, w), min(cy + win_h, h)

    roi  = frame[y1:y2, x1:x2].astype(np.float32)
    diff = np.linalg.norm(roi - avg_colour, axis=2)
    mask_local = (diff < thresh).astype(np.uint8)

    if mask_local.sum() == 0:
        return np.zeros((h, w), np.uint8), np.array([cx, cy])

    mask_full = np.zeros((h, w), np.uint8)
    mask_full[y1:y2, x1:x2] = mask_local * 255
    ys, xs = np.nonzero(mask_local)
    centroid = np.array([xs.mean() + x1, ys.mean() + y1], dtype=int)
    return mask_full, centroid

def update_running_colour(avg: np.ndarray, new_avg: np.ndarray,
                          mask_area: int, window_area: int,
                          k: int = CONFIG.MOMENTUM_K) -> np.ndarray:
    ratio = (mask_area / window_area) ** k
    return (1 - ratio) * avg + ratio * new_avg

# ───────────────────────── PLAYER POSTERISER ───────────────────────── #

def posterise_player(raw: np.ndarray, mask: np.ndarray, k: int = 4):
    x, y, w, h = cv2.boundingRect(mask)
    if w < 4 or h < 10:
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8)
    crop_rgb  = raw[y:y+h, x:x+w].copy()
    crop_mask = mask[y:y+h, x:x+w]
    crop_rgb[crop_mask == 0] = 0
    blur = cv2.bilateralFilter(crop_rgb, 5, 50, 50)
    lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    nz   = np.where(crop_mask.flatten() > 0)[0]
    if len(nz) < k:
        return crop_rgb, crop_mask
    samples = lab[nz].astype(np.float32)
    _crit, labels, centers = cv2.kmeans(samples, k, None,
                                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                        1, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    lab[nz] = centers[labels.flatten()]
    sprite = cv2.cvtColor(lab.reshape(h, w, 3), cv2.COLOR_Lab2BGR)
    return sprite, crop_mask

def paste_sprite(canvas: np.ndarray, sprite: np.ndarray,
                 alpha: np.ndarray, topleft: Tuple[int, int]) -> None:
    x, y = topleft
    h, w = alpha.shape
    roi = canvas[y:y+h, x:x+w]
    maskf = alpha.astype(float) / 255.0
    for c in range(3):
        roi[..., c] = (1 - maskf) * roi[..., c] + maskf * sprite[..., c]

# ───────────────────────── FRAME STYLISER (modified) ───────────────────────── #

def stylise(frame: np.ndarray, width: int, height: int,
            centre: Optional[Tuple[int, int]], radius: Optional[int],
            avg_col: Optional[np.ndarray], seg_model: YOLO, cfg,
            show_ball: bool) -> np.ndarray:
    """Render one frame. `show_ball=False` ⇒ no ball / no player crop."""
    drawings: list[tuple[str, object]] = []

    if show_ball and centre is not None and radius is not None and avg_col is not None:
        drawings.append(("ball", (centre, radius, avg_col)))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_pitch = cv2.inRange(hsv, *cfg.HSV_BAND_1)
    mask_pitch = cv2.morphologyEx(mask_pitch, cv2.MORPH_CLOSE, KERNEL_STRUCT, 2)
    lp = hough_lines(mask_pitch, cfg.HOUGH_1)
    if lp is not None:
        drawings.append(("pitch_lines", lp))

    mask_cross = cv2.inRange(hsv, *cfg.HSV_BAND_2)
    mask_cross = cv2.morphologyEx(mask_cross, cv2.MORPH_CLOSE, KERNEL_STRUCT, 2)
    lc = hough_lines(cv2.Canny(mask_cross, 50, 100), cfg.HOUGH_2)
    if lc is not None:
        drawings.append(("cross_lines", lc))

    # player segmentation only if we know where the ball is *and* we are showing it
    if centre is not None:
        search_half = cfg.SEARCH_SIZE // 2
        cx, cy = centre
        x1 = max(cx - search_half, 0)
        y1 = max(cy - search_half, 0)
        x2 = min(cx + search_half, width)
        y2 = min(cy + search_half, height)
        crop = frame[y1:y2, x1:x2]

        seg = seg_model(crop, task="segment",
                        conf=cfg.SEG_CONF, iou=cfg.SEG_IOU, verbose=False)[0]
        if seg.masks is not None:
            masks, classes = seg.masks.data.cpu().numpy(), seg.boxes.cls.cpu().numpy()
            scale_h = min(width, height) / cfg.BASELINE_H
            draw_r  = int(cfg.DRAW_DIST * scale_h)
            for m, cls in zip(masks, classes):
                if int(cls) != 0:
                    continue
                mb = (m * 255).astype(np.uint8)
                ch, cw = y2 - y1, x2 - x1
                if mb.shape != (ch, cw):
                    mb = cv2.resize(mb, (cw, ch), cv2.INTER_NEAREST)
                bx, by, bw, bh = cv2.boundingRect(mb)
                pcx, pcy = x1 + bx + bw // 2, y1 + by + bh // 2
                if np.hypot(pcx - cx, pcy - cy) > draw_r:
                    continue
                full = np.zeros(frame.shape[:2], np.uint8)
                full[y1:y2, x1:x2] = mb
                sprite, alpha = posterise_player(frame, full, 4)
                drawings.append(("player", (sprite, alpha, (x1 + bx, y1 + by))))

    # ─── render ───
    out = np.full_like(frame, PITCH_COLOR)

    # ─── ① Accumulate endpoints & angles ────────────────────────────── #
    vert_angles, horiz_angles = [], []

    # Anchors: None until filled
    vert_anchor = [None, None, None]          # left‑third, centre‑third, right‑third
    horiz_anchor = [None, None]               # top‑half, bottom‑half

    for k, p in drawings:
        if k != "pitch_lines":
            continue
        for x1, y1, x2, y2 in p[:, 0]:
            for (x, y) in ((x1, y1), (x2, y2)):
                dx, dy = x2 - x1, y2 - y1
                a_rad  = np.arctan2(dy, dx)
                a_deg  = np.degrees(a_rad) % 180

                #   Vertical-ish  (45°–135°)
                if 5 <= a_deg <= 175:
                    vert_angles.append(a_rad)

                    # region: 0,1,2 (left, mid, right thirds)
                    col = int(min(x, width-1) / (width/3))
                    if vert_anchor[col] is None or y < vert_anchor[col][1]:
                        vert_anchor[col] = (x, y)

                #   Horizontal-ish (else)
                else:
                    horiz_angles.append(a_rad)

                    # half: 0 = top, 1 = bottom
                    row = 0 if y < height/2 else 1

                    # choose endpoint that is closer to a screen edge
                    dist_edge = min(x, width - x)
                    if (horiz_anchor[row] is None or
                        dist_edge < horiz_anchor[row][2]):          # keep “edge‑iest”
                        horiz_anchor[row] = (x, y, dist_edge)

    # ─── ② Circular‑mean helper ─────────────────────────────────────── #
    def mean_angle(rad_list):
        if not rad_list:
            return None
        return np.arctan2(np.sum(np.sin(rad_list)), np.sum(np.cos(rad_list)))

    a_h = mean_angle(horiz_angles)
    a_v = mean_angle(vert_angles)

    line_len = max(width, height)

    # ─── ③ Draw horizontal lines (top & bottom halves) ──────────────── #
    if a_h is not None:
        for anc in horiz_anchor:
            if anc is None:
                continue
            hx, hy = anc[:2]
            # extend in both directions so it surely hits both edges
            dx, dy = np.cos(a_h), np.sin(a_h)
            p1 = (int(hx - dx*line_len), int(hy - dy*line_len))
            p2 = (int(hx + dx*line_len), int(hy + dy*line_len))
            cv2.line(out, p1, p2, GOAL_COLOR, 20)

    # ─── ④ Draw vertical lines (left/centre/right thirds) ───────────── #
    if a_v is not None:
        # always draw downward (positive‑y); flip sign if angle points up
        if np.sin(a_v) < 0:
            a_v += np.pi
        dx, dy = np.cos(a_v), np.sin(a_v)
        for anc in vert_anchor:
            if anc is None:
                continue
            vx, vy = anc
            p1 = (vx, vy)
            p2 = (int(vx + dx*line_len), int(vy + dy*line_len))
            cv2.line(out, p1, p2, GOAL_COLOR, 20)

    for k, p in drawings:
        if k == "ball":
            (cc, rr, col) = p
            cv2.circle(out, cc, rr, col.tolist(), -1)
            cv2.circle(out, cc, rr, (0, 0, 0), 2)

    crs = [p for k, p in drawings if k == "cross_lines"]
    ln = longest_valid_line(np.vstack(crs) if crs else None)
    if ln:
        (x1c, y1c), (x2c, y2c) = ln
        cv2.line(out, (x1c, y1c), (x2c, y2c), GOAL_COLOR, LINE_W)
        cv2.line(out, (x1c, y1c), (x1c, y1c + 200), GOAL_COLOR, LINE_W)
        cv2.line(out, (x2c, y2c), (x2c, y2c + 200), GOAL_COLOR, LINE_W)

    for k, p in drawings:
        if k == "player":
            sprite, alpha, tl = p
            paste_sprite(out, sprite, alpha, tl)

    return out

# ───────────────────────── MAIN PIPELINE (only ± 10 lines changed) ─────────────── #

def annotate_video(src: str | Path, cfg=CONFIG) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open {src}")

    in_fps  = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, round(in_fps / cfg.TARGET_FPS))

    out = cv2.VideoWriter(cfg.OUTPUT_NAME,
                          cv2.VideoWriter_fourcc(*'avc1'),
                          cfg.TARGET_FPS, (width, height))

    ball_yolo, seg_yolo = YOLO(cfg.BALL_MODEL), YOLO(cfg.SEG_MODEL)
    dummy = np.zeros((cfg.SEARCH_SIZE, cfg.SEARCH_SIZE, 3), np.uint8)
    ball_yolo(dummy, conf=0.01, verbose=False)
    seg_yolo(dummy, task="segment", conf=0.01, iou=0.1, verbose=False)

    buffer: list[np.ndarray] = []
    centre = radius = avg_col = None
    ball_seen = False
    frame_idx = -1

    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % interval:
            continue                           # 10 fps from the outset

        # ---------------- first‑time YOLO detection ----------------
        if not ball_seen:
            res = ball_yolo(fr, conf=cfg.BALL_CONF, verbose=False)[0]
            for box in res.boxes:
                if ball_yolo.names[int(box.cls[0])] != cfg.BALL_LABEL:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centre = ((x1 + x2)//2, (y1 + y2)//2)
                radius = ((x2 - x1) + (y2 - y1)) // 4
                m = np.zeros(fr.shape[:2], np.uint8)
                cv2.circle(m, centre, radius, 255, -1)
                avg_col = np.array(cv2.mean(fr, mask=m)[:3])
                ball_seen = True
                break

        if not ball_seen:          # still waiting → cache frame & continue
            buffer.append(fr.copy())
            continue

        # ---------------- backward colour‑tracking pass ------------
        if buffer:
            for prev in reversed(buffer):
                msk, new_c = vectorised_colour_search(
                    prev, centre, (width//10, height//10),
                    avg_col, cfg.COLOUR_THRESH)
                if msk.any():
                    new_avg = np.array(cv2.mean(prev, mask=msk)[:3])
                    avg_col = update_running_colour(avg_col, new_avg,
                                                   msk.sum()//255,
                                                   (width//10)*(height//10))
                    centre = tuple(new_c)
                    show_ball = True
                else:
                    show_ball = False
                out_prev = stylise(prev, width, height,
                                   centre, radius, avg_col,
                                   seg_yolo, cfg, show_ball)
                out.write(out_prev)
            buffer.clear()

        # ---------------- forward processing -----------------------
        show_ball = False
        if centre is not None:
            msk, new_c = vectorised_colour_search(
                fr, centre, (width//10, height//10),
                avg_col, cfg.COLOUR_THRESH)
            if msk.any():
                new_avg = np.array(cv2.mean(fr, mask=msk)[:3])
                avg_col = update_running_colour(avg_col, new_avg,
                                               msk.sum()//255,
                                               (width//10)*(height//10))
                centre = tuple(new_c)
                show_ball = True

        out_f = stylise(fr, width, height,
                        centre, radius, avg_col,
                        seg_yolo, cfg, show_ball)
        out.write(out_f)

    cap.release()
    out.release()
    print(f"Finished → {cfg.OUTPUT_NAME}")

# ───────────────────────── CLI ───────────────────────── #
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "TestGoal.mp4"
    annotate_video(src)