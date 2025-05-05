# config.py

# --- Module switch ---
USE_KALMAN = True          # Whether to enable Kalman filter
USE_REID = True            # Whether to enable Re-ID

# --- Model parameter ---
YOLO_MODEL_VERSION = 'yolo11n.pt'
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.7
SORT_IOU_THRESHOLD = 0.2
SORT_MIN_HITS = 3
REID_SIM_THRESHOLD = 0.1

# --- Video ---
VIDEO_PATH = 'input/vehicle_night.mp4'
OUTPUT_VIDEO_PATH = 'output/yolo_kalman_reid/vehicle_night.mp4'
SAVE_VIDEO = True

# --- Run Performance Settings ---
DISPLAY = True             # Whether to display the window
FRAME_SKIP = 1             # Run every few frames (increases speed)
