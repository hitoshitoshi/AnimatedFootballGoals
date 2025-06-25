import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO

from .config import CONFIG
from .vision import combine_lines_geom, hough_lines, longest_valid_line

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
    mask_pitch = cv2.morphologyEx(mask_pitch, cv2.MORPH_CLOSE, CONFIG.KERNEL_STRUCT, 2)
    lp = hough_lines(mask_pitch, cfg.HOUGH_1)
    if lp is not None:
        merged = combine_lines_geom(lp)
        drawings.append(("pitch_lines", merged))

    # Detect crossbar lines similarly
    mask_cross = cv2.inRange(hsv, *cfg.HSV_BAND_2)
    mask_cross = cv2.morphologyEx(mask_cross, cv2.MORPH_CLOSE, CONFIG.KERNEL_STRUCT, 2)
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
    out = np.full_like(frame, CONFIG.PITCH_COLOR)

    # Draw pitch lines
    for kind, data in drawings:
        if kind == "pitch_lines":
            for x1,y1,x2,y2 in data:
                cv2.line(out, (x1,y1), (x2,y2), CONFIG.GOAL_COLOR, CONFIG.LINE_W)

    # Draw goal crossbar and posts
    crs = [d for k,d in drawings if k=="cross_lines"]
    ln = longest_valid_line(np.vstack(crs) if crs else None)
    if ln:
        (x1c,y1c),(x2c,y2c) = ln
        # crossbar
        cv2.line(out,(x1c,y1c),(x2c,y2c),CONFIG.GOAL_COLOR,CONFIG.LINE_W)
        # left post
        cv2.line(out,(x1c,y1c),(x1c,y1c+375),CONFIG.GOAL_COLOR,CONFIG.LINE_W)
        # right post
        cv2.line(out,(x2c,y2c),(x2c,y2c+425),CONFIG.GOAL_COLOR,CONFIG.LINE_W)

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