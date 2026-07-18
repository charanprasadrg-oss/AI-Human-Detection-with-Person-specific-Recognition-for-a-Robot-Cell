"""
Simplified AI Human Detection with Person-Specific Recognition
--------------------------------------------------------------

- Uses YOLOv8 (ultralytics) + OpenCV.
- Detects persons in frames (webcam or video).
- Checks if they are inside a safety zone.
- For persons in the safety zone:
    - Analyses "jacket region" and estimates how black it is.
    - If jacket is black  -> authorized -> Robot RUN.
    - Else / low confidence / no person in zone -> Robot STOP (fail-safe).
- Logs events to safety_log.csv (Excel-compatible).
- Draws bounding boxes, safety zone, and robot state text.

This version is:
- Simple, portable, and can be run in any normal Python environment.
- For Jetson Nano you only need to change the camera source function.
"""

import cv2
import csv
import os
import time
from datetime import datetime

import numpy as np
from ultralytics import YOLO


# =========================
# CONFIGURATION
# =========================

# 1) MODEL PATH
# Put your Roboflow-exported YOLOv8 model here (e.g. "best.pt").
# For quick online testing, you can also use a built-in model like "yolov8n.pt",
# but that will detect many classes, not only persons.
YOLO_MODEL_PATH = "yolov8n.pt"   # change to "weights/best.pt" in your project

# 2) VIDEO SOURCE
# For online compilers without webcam you may need to upload a video and
# change VIDEO_SOURCE to that file name.
# - 0 means default webcam
# - "video.mp4" means a video file
VIDEO_SOURCE = 0  # change to "robot_cell_demo.mp4" for file

# 3) SAFETY ZONE (rectangle)
# You can tune these per scene. They are in pixel coordinates (x1, y1, x2, y2).
SAFETY_ZONE = {
    "x1": 200,
    "y1": 100,
    "x2": 800,
    "y2": 600
}

# 4) DETECTION THRESHOLDS
CONFIDENCE_THRESHOLD = 0.4   # YOLO confidence threshold for detections
IOU_THRESHOLD = 0.45         # IoU threshold for NMS
MIN_CONF_FOR_RUN = 0.6       # fail-safe min confidence

# 5) JACKET COLOUR (BLACK) DETECTION
BLACK_THRESHOLD = 60         # B,G,R < this => considered black pixel
JACKET_TOP_REL = 0.3        # relative top of jacket region in person bbox
JACKET_BOTTOM_REL = 0.8     # relative bottom of jacket region in person bbox
JACKET_BLACK_RATIO_THRESHOLD = 0.4  # >= 40% black pixels -> "black jacket"

# 6) LOG FILE
LOG_FILE = "safety_log.csv"


# =========================
# UTILITY FUNCTIONS
# =========================

def init_log_file(path: str):
    """Create CSV log file with header if it does not exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "robot_state",   # RUN / STOP
                "reason",        # no_person / unauthorized_person / low_confidence / authorized_persons
                "num_persons_in_zone",
                "max_confidence"
            ])


def log_event(path: str, robot_state: str, reason: str,
              num_persons: int, max_conf: float):
    """Append event to CSV log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, robot_state, reason, num_persons, f"{max_conf:.3f}"])


def does_box_intersect_zone(x1: int, y1: int, x2: int, y2: int, zone: dict) -> bool:
    """Return True if [x1,y1,x2,y2] bounding box intersects the safety zone."""
    ix1 = max(x1, zone["x1"])
    iy1 = max(y1, zone["y1"])
    ix2 = min(x2, zone["x2"])
    iy2 = min(y2, zone["y2"])
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


def compute_jacket_black_ratio(person_crop: np.ndarray) -> float:
    """
    Compute fraction of black pixels in the "jacket region" of a person crop.
    - Input: BGR image of detected person.
    - Output: ratio in [0,1].
    """
    if person_crop is None or person_crop.size == 0:
        return 0.0

    h, w, _ = person_crop.shape
    top = int(h * JACKET_TOP_REL)
    bottom = int(h * JACKET_BOTTOM_REL)

    if bottom <= top or top < 0 or bottom > h:
        return 0.0

    jacket_region = person_crop[top:bottom, :]
    if jacket_region.size == 0:
        return 0.0

    b, g, r = cv2.split(jacket_region)
    black_mask = (b < BLACK_THRESHOLD) & (g < BLACK_THRESHOLD) & (r < BLACK_THRESHOLD)
    black_pixels = np.count_nonzero(black_mask)
    total_pixels = jacket_region.shape[0] * jacket_region.shape[1]
    if total_pixels == 0:
        return 0.0

    return black_pixels / float(total_pixels)


def is_authorized_jacket(black_ratio: float) -> bool:
    """Return True if jacket is considered 'black' based on ratio threshold."""
    return black_ratio >= JACKET_BLACK_RATIO_THRESHOLD


# =========================
# MAIN
# =========================

def main():
    # 1) Init log
    init_log_file(LOG_FILE)

    # 2) Load YOLO model
    print(f"[INFO] Loading YOLOv8 model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # 3) Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    # 4) Robot state
    robot_state = "RUN"

    cv2.namedWindow("Robot Cell Safety", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video / no frame.")
            break

        start = time.time()
        persons_in_zone = []
        max_conf = 0.0

        # --- YOLO inference ---
        # We ask YOLO to do detection on the frame.
        results = model(frame, conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # By default yolov8n.pt has class 0 = 'person'
                if result.names[cls_id].lower() != "person":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if conf > max_conf:
                    max_conf = conf

                in_zone = does_box_intersect_zone(x1, y1, x2, y2, SAFETY_ZONE)

                # Draw box
                color = (0, 255, 0) if in_zone else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"person {conf:.2f}",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if in_zone:
                    # Crop person and compute black ratio
                    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                    black_ratio = compute_jacket_black_ratio(crop)
                    authorized = is_authorized_jacket(black_ratio)

                    persons_in_zone.append({
                        "conf": conf,
                        "black_ratio": black_ratio,
                        "authorized": authorized
                    })

                    # Draw jacket region rectangle
                    h = y2 - y1
                    j_top = int(y1 + JACKET_TOP_REL * h)
                    j_bottom = int(y1 + JACKET_BOTTOM_REL * h)
                    jacket_color = (255, 0, 0) if authorized else (0, 0, 255)
                    cv2.rectangle(frame, (x1, j_top), (x2, j_bottom), jacket_color, 1)
                    cv2.putText(frame, f"black={black_ratio:.2f}",
                                (x1, j_bottom + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # --- Safety decision logic (fail-safe) ---
        new_state = "RUN"
        reason = "no_person"

        if len(persons_in_zone) == 0:
            new_state = "RUN"
            reason = "no_person"
        else:
            any_unauthorized = any(not p["authorized"] for p in persons_in_zone)
            low_conf = max_conf < MIN_CONF_FOR_RUN

            if any_unauthorized or low_conf:
                new_state = "STOP"
                reason = "unauthorized_person" if any_unauthorized else "low_confidence"
            else:
                new_state = "RUN"
                reason = "authorized_persons"

        # Log if state changed
        if new_state != robot_state:
            robot_state = new_state
            log_event(LOG_FILE, robot_state, reason, len(persons_in_zone), max_conf)
            print(f"[EVENT] Robot={robot_state}, reason={reason}, "
                  f"persons_in_zone={len(persons_in_zone)}, max_conf={max_conf:.2f}")

        # --- Draw safety zone and state ---
        cv2.rectangle(frame,
                      (SAFETY_ZONE["x1"], SAFETY_ZONE["y1"]),
                      (SAFETY_ZONE["x2"], SAFETY_ZONE["y2"]),
                      (0, 0, 255), 2)

        state_color = (0, 255, 0) if robot_state == "RUN" else (0, 0, 255)
        cv2.putText(frame, f"Robot: {robot_state}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    state_color,
                    2,
                    cv2.LINE_AA)

        dt = time.time() - start
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Robot Cell Safety", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished. Events logged in:", LOG_FILE)


