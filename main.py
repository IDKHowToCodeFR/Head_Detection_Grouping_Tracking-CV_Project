"""
Automated Head Detection and Group Tracking Pipeline
---------------------------------------------------
This script combines all stages:
1.  Detection:   Runs YOLO to detect heads and create "dot videos".
2.  Grouping:    Analyzes the first frame of dot videos to find initial groups.
3.  Tracking:    Uses optical flow to track groups and validates their cohesion.

To Run:
1.  Place videos in the '0_input_videos' folder.
2.  Place 'best.pt' model in the 'models' folder.
3.  Run: `python main.py`
"""

import os
import cv2
import numpy as np
import json
import glob
from pathlib import Path
from ultralytics import YOLO

# Import configuration and utility functions
import config as cfg
import utils


def ensure_output_dirs_exist():
    """Creates all output directories defined in the config if they don't exist."""
    print("Ensuring output directories exist...")
    for dir_path in cfg.ALL_OUTPUT_DIRS:
        os.makedirs(dir_path, exist_ok=True)
    print("Output directories are ready.")


# --- STAGE 1: DETECTION ---

def run_stage_1_detection(video_path: Path) -> Path:
    """
    Loads the YOLO model and runs inference on a single video.
    Outputs a new video showing only a green dot for each detected person.

    Returns the path to the created dot video.
    """
    print(f"\n--- STAGE 1: Starting Detection for {video_path.name} ---")

    model_path = cfg.MODEL_DIR / cfg.MODEL_NAME
    if not model_path.exists():
        print(f"Error: Model weights not found at '{model_path}'.")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(model_path)

    # Use streaming for efficient video processing
    results_stream = model.predict(
        source=str(video_path),
        conf=cfg.DETECTION_CONFIDENCE,
        imgsz=cfg.IMAGE_SIZE,
        save=False,
        verbose=False,
        stream=True
    )

    # Prepare output video
    output_filename = f"{video_path.stem}_dots.avi"
    output_filepath = cfg.STAGE_1_DOT_VIDEOS_DIR / output_filename

    writer = None

    for result in results_stream:
        im_bgr = result.orig_img

        if writer is None:
            # Initialize VideoWriter on the first frame
            height, width = im_bgr.shape[:2]
            fps = result.fps if hasattr(result, 'fps') and result.fps else 30
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(output_filepath), fourcc, fps, (width, height))
            print(f"Writing dot video to: {output_filepath}")

        # Create a black frame to draw dots on
        dot_frame = np.zeros_like(im_bgr)

        # Draw dots for each detected person
        for box in result.boxes:
            if box.conf[0].item() >= cfg.DETECTION_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(dot_frame, (center_x, center_y),
                           radius=cfg.DOT_RADIUS,
                           color=cfg.DOT_COLOR,
                           thickness=cfg.DOT_THICKNESS)

        writer.write(dot_frame)

    if writer:
        writer.release()

    print(f"--- STAGE 1: Finished Detection for {video_path.name} ---")
    return output_filepath


# --- STAGE 2: INITIAL GROUPING ---

def find_connected_groups(dot_centers: list[tuple[int, int]], radius: int) -> list[list[int]]:
    """
    Identifies groups of connected dots using Breadth-First Search (BFS).
    Two dots are connected if the distance between them is <= 2 * radius.
    """
    if not dot_centers:
        return []

    num_dots = len(dot_centers)
    adj = {i: [] for i in range(num_dots)}

    for i in range(num_dots):
        for j in range(i + 1, num_dots):
            dist = np.linalg.norm(np.array(dot_centers[i]) - np.array(dot_centers[j]))
            if dist <= 2 * radius:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    groups = []
    for i in range(num_dots):
        if i not in visited:
            current_group_indices = []
            q = [i]
            visited.add(i)
            while q:
                u = q.pop(0)
                current_group_indices.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            groups.append(current_group_indices)
    return groups


def run_stage_2_grouping(dot_video_path: Path) -> Path:
    """
    Extracts the first frame of the dot video, identifies groups,
    and saves the initial group data to a JSON file.
    Also saves a visualization of the first frame.

    Returns the path to the created JSON file.
    """
    print(f"\n--- STAGE 2: Starting Initial Grouping for {dot_video_path.name} ---")

    cap = cv2.VideoCapture(str(dot_video_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video {dot_video_path}")
        return

    success, frame = cap.read()
    if not success:
        print(f"Error: Failed to read first frame from {dot_video_path}")
        cap.release()
        return
    cap.release()

    # --- Dot Detection and Grouping ---
    dot_centers = utils.detect_dots_in_frame(frame)
    num_dots = len(dot_centers)
    radius = utils.get_radius(num_dots)
    groups = find_connected_groups(dot_centers, radius)

    print(f"Found {num_dots} dots, forming {len(groups)} groups. Radius: {radius}")

    # --- Data Collection and Drawing ---
    output_image = frame.copy()
    json_output_data = []

    for i, group_indices in enumerate(groups):
        group_id = i + 1
        group_centers = [dot_centers[idx] for idx in group_indices]

        json_output_data.append({
            "group_id": group_id,
            "group_members": len(group_centers),
            "member_locations": group_centers
        })

        # Draw connecting lines
        for j1 in range(len(group_centers)):
            for j2 in range(j1 + 1, len(group_centers)):
                p1 = group_centers[j1]
                p2 = group_centers[j2]
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist <= 2 * radius:
                    cv2.line(output_image, p1, p2, cfg.GROUP_LINE_COLOR, thickness=3)

        # Draw circles for the dots
        for center in group_centers:
            cv2.circle(output_image, center, radius, cfg.GROUP_CIRCLE_COLOR, thickness=2)

        # Draw Group ID
        if group_centers:
            avg_x = int(np.mean([p[0] for p in group_centers]))
            avg_y = int(np.mean([p[1] for p in group_centers]))
            font_scale = 0.5 + (radius / 100.0)
            thickness = max(1, int(font_scale * 2))
            cv2.putText(output_image, f"G{group_id}", (avg_x, avg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, cfg.GROUP_TEXT_COLOR,
                        thickness, cv2.LINE_AA)

    # --- Save Outputs ---
    base_name = dot_video_path.stem

    # Save visualization image
    output_image_path = cfg.STAGE_2_INIT_FRAMES_DIR / f"{base_name}_groups.jpg"
    cv2.imwrite(str(output_image_path), output_image)
    print(f"Saved initial group visualization to: {output_image_path}")

    # Save group data to JSON
    output_json_path = cfg.STAGE_2_INIT_FRAMES_DIR / f"{base_name}_groups.json"
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=4)
    print(f"Saved initial group data to: {output_json_path}")

    print(f"--- STAGE 2: Finished Initial Grouping for {dot_video_path.name} ---")
    return output_json_path


# --- STAGE 3: TRACKING & VALIDATION ---

def draw_tracked_groups(frame: np.ndarray, groups: list, radius: int) -> np.ndarray:
    """Draws tracked groups, connecting lines, and distances on the frame."""
    for group in groups:
        group_id = group["group_id"]
        group_centers = group["member_locations"]

        # Draw lines and distances
        for i in range(len(group_centers)):
            for j in range(i + 1, len(group_centers)):
                p1 = tuple(map(int, group_centers[i]))
                p2 = tuple(map(int, group_centers[j]))
                cv2.line(frame, p1, p2, cfg.GROUP_LINE_COLOR, thickness=3)

                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2) - 5)
                cv2.putText(frame, f"{int(dist)}", mid_point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw circles
        for center_float in group_centers:
            center = tuple(map(int, center_float))
            cv2.circle(frame, center, radius, cfg.GROUP_CIRCLE_COLOR, thickness=2)

        # Draw Group ID
        if group_centers:
            avg_x = int(np.mean([p[0] for p in group_centers]))
            avg_y = int(np.mean([p[1] for p in group_centers]))
            font_scale = 0.5 + (radius / 100.0)
            thickness = max(1, int(font_scale * 2))
            cv2.putText(frame, f"G{group_id}", (avg_x, avg_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, cfg.GROUP_TEXT_COLOR, thickness, cv2.LINE_AA)
    return frame


def calculate_internal_distances(members: list) -> list[dict]:
    """Calculates the shortest distance to a neighbor for each member."""
    num_members = len(members)
    output_members = []
    if num_members == 0:
        return []

    for i in range(num_members):
        p1 = np.array(members[i])
        min_dist = float('inf')
        if num_members > 1:
            for j in range(num_members):
                if i == j: continue
                p2 = np.array(members[j])
                dist = np.linalg.norm(p1 - p2)
                if dist < min_dist:
                    min_dist = dist
        else:
            min_dist = 0  # Single member group

        output_members.append({
            "location": members[i],
            "shortest_dist": round(min_dist, 2)
        })
    return output_members


def validate_group_cohesion(groups: list, threshold: float, frame_count: int, log_file: 'TextIOWrapper'):
    """Logs a warning if a member is further than the threshold from its nearest neighbor."""
    for group in groups:
        group_id = group['group_id']
        members = group['member_locations']
        num_members = len(members)
        if num_members <= 1:
            continue

        for i in range(num_members):
            p1 = np.array(members[i])
            min_dist = float('inf')
            for j in range(num_members):
                if i == j: continue
                dist = np.linalg.norm(p1 - np.array(members[j]))
                if dist < min_dist:
                    min_dist = dist

            if min_dist > threshold:
                log_message = (f"Frame {frame_count}: Group {group_id} member at {tuple(map(int, p1))} "
                               f"is lost. Min distance: {min_dist:.2f} > {threshold}")
                print(log_message)
                log_file.write(log_message + "\n")


def run_stage_3_tracking(dot_video_path: Path, initial_groups_json_path: Path):
    """
    Tracks the initial groups using Lucas-Kanade optical flow,
    validates group cohesion, and removes stationary groups.
    """
    print(f"\n--- STAGE 3: Starting Tracking for {dot_video_path.name} ---")

    # --- Initialization ---
    cap = cv2.VideoCapture(str(dot_video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {dot_video_path}");
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    stationary_threshold_frames = int(cfg.STATIONARY_THRESHOLD_SECONDS * fps)
    stationary_tracker = {}  # {group_id: {'last_centroid': (x, y), 'frames_still': 0}}

    base_name = dot_video_path.stem
    output_video_path = cfg.STAGE_3_TRACKED_VIDEOS_DIR / f"{base_name}_tracked.avi"
    output_json_path = cfg.STAGE_3_TRACKED_VIDEOS_DIR / f"{base_name}_final_groups.json"
    log_file_path = cfg.STAGE_3_LOGS_DIR / f"{base_name}_validation.log"

    log_file = open(log_file_path, 'w')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, int(fps),
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # --- First Frame Processing ---
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Cannot read video file {dot_video_path}");
        log_file.close();
        return

    tracked_groups = utils.load_initial_groups(initial_groups_json_path)
    if tracked_groups is None:
        log_file.close();
        return

    num_initial_dots = sum(g['group_members'] for g in tracked_groups)
    radius = utils.get_radius(num_initial_dots)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_dots = np.array([loc for group in tracked_groups for loc in group['member_locations']],
                         dtype=np.float32).reshape(-1, 1, 2)

    output_frame = draw_tracked_groups(first_frame.copy(), tracked_groups, radius)
    out_writer.write(output_frame)
    frame_count = 1

    # --- Subsequent Frame Processing ---
    while True:
        ret, frame = cap.read()
        if not ret: break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_dots is not None and len(prev_dots) > 0:
            new_dots, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_dots, None,
                                                           **cfg.LK_PARAMS)
            good_new = new_dots[status == 1]
            good_old_indices = np.where(status == 1)[0]
        else:
            good_new, good_old_indices = np.array([]), []

        # Update group locations based on flow
        dot_idx_counter = 0
        for group in tracked_groups:
            member_count = group['group_members']
            tracked_indices = [i for i, old_idx in enumerate(good_old_indices) if
                               dot_idx_counter <= old_idx < dot_idx_counter + member_count]
            group['member_locations'] = [good_new[i].ravel().tolist() for i in tracked_indices]
            group['group_members'] = len(group['member_locations'])
            dot_idx_counter += member_count

        tracked_groups = [g for g in tracked_groups if g['group_members'] > 0]

        # --- Stillness Detection ---
        groups_to_remove = set()
        for group in tracked_groups:
            if not group['member_locations']: continue
            group_id = group['group_id']
            current_centroid = tuple(np.mean(group['member_locations'], axis=0))

            if group_id in stationary_tracker:
                last_centroid = stationary_tracker[group_id]['last_centroid']
                if np.linalg.norm(
                        np.array(current_centroid) - np.array(last_centroid)) < cfg.STATIONARY_PIXEL_THRESHOLD:
                    stationary_tracker[group_id]['frames_still'] += 1
                else:
                    stationary_tracker[group_id]['frames_still'] = 0
                    stationary_tracker[group_id]['last_centroid'] = current_centroid
            else:
                stationary_tracker[group_id] = {'last_centroid': current_centroid, 'frames_still': 0}

            if stationary_tracker[group_id]['frames_still'] > stationary_threshold_frames:
                groups_to_remove.add(group_id)

        if groups_to_remove:
            print(f"Removing stationary groups: {groups_to_remove}")
            log_file.write(f"Frame {frame_count}: Removing stationary groups: {groups_to_remove}\n")
            tracked_groups = [g for g in tracked_groups if g['group_id'] not in groups_to_remove]
            for group_id in groups_to_remove:
                if group_id in stationary_tracker:
                    del stationary_tracker[group_id]

        # --- Group Cohesion Validation ---
        validate_group_cohesion(tracked_groups, cfg.MAX_MEMBER_DISTANCE_THRESHOLD, frame_count, log_file)

        # Draw and write frame
        output_frame = draw_tracked_groups(frame.copy(), tracked_groups, radius)
        out_writer.write(output_frame)

        # Update previous frame and points
        prev_gray = current_gray.copy()
        prev_dots = np.array([loc for group in tracked_groups for loc in group['member_locations']],
                             dtype=np.float32).reshape(-1, 1, 2)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")

    # --- Finalization ---
    final_output_data = []
    for group in tracked_groups:
        final_output_data.append({
            "group_id": group.get('group_id'),
            "group_members": len(group.get('member_locations', [])),
            "members": calculate_internal_distances(group.get('member_locations', []))
        })

    with open(output_json_path, 'w') as f:
        json.dump(final_output_data, f, indent=4)

    log_file.close()
    cap.release()
    out_writer.release()

    print(f"\nTracking complete for {dot_video_path.name}")
    print(f"  -> Tracked video saved: {output_video_path}")
    print(f"  -> Final group JSON saved: {output_json_path}")
    print(f"  -> Validation log saved: {log_file_path}")
    print(f"--- STAGE 3: Finished Tracking ---")


# --- MAIN ORCHESTRATOR ---

def main():
    """
    Main function to run the complete program.
    """
    print("========================================")
    print("   Starting Head & Group Track Pipeline   ")
    print("========================================")

    ensure_output_dirs_exist()

    # Find all videos in the source directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    source_videos = []
    for ext in video_extensions:
        source_videos.extend(glob.glob(str(cfg.SOURCE_VIDEO_DIR / f"*{ext}")))

    if not source_videos:
        print(f"Error: No videos found in {cfg.SOURCE_VIDEO_DIR}")
        print("Please add your videos to the '0_input_videos' folder and try again.")
        return

    print(f"Found {len(source_videos)} video(s) to process:")
    for video in source_videos:
        print(f"  - {Path(video).name}")

    # Process each video through the full pipeline
    for video_file in source_videos:
        video_path = Path(video_file)
        print(f"\nProcessing: {video_path.name}...")

        try:
            # Stage 1: Detection
            dot_video_path = run_stage_1_detection(video_path)

            # Stage 2: Initial Grouping
            initial_groups_json_path = run_stage_2_grouping(dot_video_path)

            # Stage 3: Tracking & Validation
            run_stage_3_tracking(dot_video_path, initial_groups_json_path)

            print(f"\nSuccessfully completed processing for {video_path.name}")

        except Exception as e:
            print(f"!!!!!!!!!! ERROR processing {video_path.name} !!!!!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {video_path.name} and moving to the next video.")

    print("\n========================================")
    print("   Pipeline processing complete.   ")
    print("========================================")


if __name__ == '__main__':
    main()