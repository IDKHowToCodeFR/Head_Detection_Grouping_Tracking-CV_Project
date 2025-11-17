"""
Utility Functions
-----------------
Reusable helper functions imported by the main file.
This avoids code duplication.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import config as cfg


def get_radius(num_dots: int) -> int:
    """
    Calculates a dynamic radius based on the number of dots.
    Uses the formula and parameters from the config file.
    """
    if num_dots <= 0:
        return 0

    radius = (cfg.GROUPING_K_FACTOR / (num_dots ** cfg.GROUPING_ALPHA)) * \
             np.log(cfg.GROUPING_BETA * num_dots + 1)

    clamped_radius = max(cfg.GROUPING_MIN_RADIUS, min(radius, cfg.GROUPING_MAX_RADIUS))
    return int(clamped_radius)


def detect_dots_in_frame(frame: np.ndarray) -> list[tuple[int, int]]:
    """
    Finds all green dots in a given frame based on the BGR range in config.
    Returns a list of (x, y) center coordinates.
    """
    mask = cv2.inRange(frame, cfg.GREEN_DOT_BGR_LOWER, cfg.GREEN_DOT_BGR_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dot_centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dot_centers.append((cX, cY))
    return dot_centers


def load_initial_groups(json_path: Path) -> list | None:
    """
    Loads the initial group data from a JSON file.
    """
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return None