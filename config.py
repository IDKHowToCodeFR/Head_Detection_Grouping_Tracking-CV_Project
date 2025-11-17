"""
Central Configuration File
--------------------------
All file paths, model names, and algorithm parameters are defined here.
This makes the pipeline portable and easy to tune.
"""
from pathlib import Path

# --- Core Paths ---
# The base directory of the project (where this config.py file is)
BASE_DIR = Path(__file__).resolve().parent

# --- Input Folders ---
# Place your input videos in this folder
SOURCE_VIDEO_DIR = BASE_DIR / "0_input_videos"
# Place your YOLO model weights here
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "best.pt"

# --- Output Folders (The pipeline will create these) ---
OUTPUT_DIR = BASE_DIR / "output"
STAGE_1_DOT_VIDEOS_DIR = OUTPUT_DIR / "1_dot_videos"
STAGE_2_INIT_FRAMES_DIR = OUTPUT_DIR / "2_init_frames"
STAGE_3_TRACKED_VIDEOS_DIR = OUTPUT_DIR / "3_tracked_videos"
STAGE_3_LOGS_DIR = OUTPUT_DIR / "4_validation_logs"

# List of all output directories for automated creation
ALL_OUTPUT_DIRS = [
    OUTPUT_DIR,
    STAGE_1_DOT_VIDEOS_DIR,
    STAGE_2_INIT_FRAMES_DIR,
    STAGE_3_TRACKED_VIDEOS_DIR,
    STAGE_3_LOGS_DIR
]

# --- Stage 1: Detection Parameters ---
DETECTION_CONFIDENCE = 0.25
IMAGE_SIZE = 640
# Green dot color (BGR)
DOT_COLOR = (0, 255, 0)
DOT_RADIUS = 2
DOT_THICKNESS = -1  # -1 for a filled circle

# --- Stage 2: Initial Grouping Parameters ---
# Color range to detect the green dots (BGR)
GREEN_DOT_BGR_LOWER = (0, 150, 0)
GREEN_DOT_BGR_UPPER = (100, 255, 100)

# Dynamic radius calculation for grouping
# Formula: r = (K / (num_dots^ALPHA)) * log(BETA * num_dots + 1)
GROUPING_MIN_RADIUS = 5
GROUPING_MAX_RADIUS = 300
GROUPING_K_FACTOR = 500
GROUPING_ALPHA = 0.9
GROUPING_BETA = 0.17

# Visualization colors (BGR)
GROUP_LINE_COLOR = (128, 0, 128)  # Purple
GROUP_CIRCLE_COLOR = (0, 0, 255)   # Red
GROUP_TEXT_COLOR = (255, 255, 0) # Cyan

# --- Stage 3: Tracking Parameters ---
# Lucas-Kanade Optical Flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(1 | 2, 10, 0.03)  # (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# How long a group must be still (in seconds) to be removed
STATIONARY_THRESHOLD_SECONDS = 2.0
# Pixel distance movement threshold to be considered "still"
STATIONARY_PIXEL_THRESHOLD = 1.0

# Max allowed distance (in pixels) between a member and its nearest neighbor
# If exceeded, the member is flagged as "lost"
MAX_MEMBER_DISTANCE_THRESHOLD = 150