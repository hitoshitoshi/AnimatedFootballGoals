import numpy as np
import cv2, numpy as np, sys

class CONFIG:
    """Configuration parameters for video processing and detection."""
    # Output video settings
    OUTPUT_NAME = "./outputs/output.mp4"
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

    KERNEL_STRUCT = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL)