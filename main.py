import cv2

def downsample_video(input_path, output_path, target_fps=10):
    """
    Reads `input_path` at its native frame rate, drops frames so the output 
    has `target_fps`, and writes to `output_path` (mp4).

    :param input_path: Path to the original video.
    :param output_path: Path for the downsampled output video.
    :param target_fps: Desired frames per second for the output.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    # Get the original FPS and dimensions
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate how many frames we skip between each frame we keep
    # (rounding to ensure we don't miss frames due to float precision)
    frame_interval = int(round(original_fps / target_fps))

    # Define output codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_index = 0
    written_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write only every "frame_interval"-th frame
        if frame_index % frame_interval == 0:
            out.write(frame)
            written_frames += 1
        
        frame_index += 1

    cap.release()
    out.release()
    print(f"Downsampled video saved at {output_path}.")
    print(f"Original FPS: {original_fps:.2f}, New FPS: ~{target_fps}, Total Frames Written: {written_frames}")

downsample_video("TestGoal.mp4", "cartoon_output.mp4")