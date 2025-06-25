# Animated Football Goals

This project automatically processes football match videos to create stylized, animated representations of the game. It uses computer vision techniques to detect and track the ball, players, and pitch lines, and then redraws them in a simplified, cartoon-like aesthetic.

## Features

  * **Pitch and Goal Detection:** Identifies the pitch lines and goalposts from the video using color filtering and Hough transforms.
  * **Ball Tracking:** Employs a YOLOv8 model for initial ball detection, followed by a color-based tracking algorithm to follow the ball's movement.
  * **Player Segmentation and Stylization:** Detects players near the ball using a YOLO segmentation model and renders them as posterized sprites, preserving their general outline and colors.
  * **Animated Output:** Combines all the detected elements into a final animated video with a clean, stylized look.

## How It Works

The main script (`main.py`) orchestrates the entire process:

1.  **Video Loading and Preprocessing:** The input video is loaded, and its frame rate is adjusted to a target FPS for consistent processing.
2.  **Ball Detection:** The script first finds the initial position of the ball using a pre-trained YOLO model (`yolo11x.pt`).
3.  **Ball and Player Tracking:** In subsequent frames, the ball's position is tracked using its color and motion. A region of interest (ROI) around the ball is analyzed to detect and segment players using a YOLO segmentation model (`yolo11x-seg.pt`).
4.  **Frame Stylization:** Each frame is reconstructed with a plain green background representing the pitch. Detected pitch lines and goalposts are drawn in white. Players are rendered as simplified, posterized versions of their real-life counterparts.
5.  **Video Output:** The stylized frames are compiled into a new MP4 video file named `output.mp4` inside the `outputs` folder.

## Getting Started

### Prerequisites

You will need Python 3 and the libraries listed in the `requirements.txt` file.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/hitoshitoshi/AnimatedFootballGoals.git
    cd AnimatedFootballGoals
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Place your input video file in the `data` directory (you may need to create this directory). The project is configured to look for `TestGoal.mp4` by default.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  The processed video will be saved as `outputs/output.mp4`.

You can also run the script with a specific video file as a command-line argument:

```bash
python main.py /path/to/your/video.mp4
```

## Project Structure

  * `main.py`: The main entry point for the application.
  * `animator/`: A directory containing the core logic for the animation process.
      * `config.py`: Configuration file for various parameters like model paths, confidence thresholds, and color values.
      * `tracking.py`: Functions for tracking the ball based on color.
      * `stylization.py`: Code for rendering the stylized frames, including players and pitch elements.
      * `vision.py`: Helper functions for computer vision tasks like line detection and merging.
  * `notebooks/`: Contains a Jupyter notebook (`main.ipynb`) for experimenting with the different components of the project.
  * `requirements.txt`: A list of the Python packages required to run the project.
  * `.gitignore`: Specifies which files and directories to ignore in version control.
