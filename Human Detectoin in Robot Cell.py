"""
AI Human Detection with Person-Specific Recognition for a Robot Cell
-------------------------------------------------------------------

- Uses YOLOv8 (ultralytics) + OpenCV (cv2).
- Detects persons in video frames (webcam or video file).
- Checks if they intersect a defined safety zone.
- For persons in the safety zone:
    - Analyses jacket region and estimates how black it is.
    - Black jacket -> authorized -> Robot RUN.
    - Other / low confidence / none -> Robot STOP (fail-safe).
- Logs RUN/STOP events to safety_log.csv.

This version uses the built-in 'yolov8n.pt' model by default.
To use your Roboflow-trained model later, change YOLO_MODEL_PATH.
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

# Path to YOLOv8 model
# For now, use the default ultralytics model (downloaded automatically).
# Later, replace this with "weights/best.pt" when you add your Roboflow model.
YOLO_MODEL_PATH = "yolov8n.pt"

# Video source:
#   0             -> default webcam
#   "video.mp4"   -> video file
VIDEO_SOURCE = 0  # change to "data/robot_cell_demo.mp4" if you have a video

# Safety zone (rectangle in image coordinates)
SAFETY_ZONE = {
    "x1": 200,
    "y1": 100,
    "x2": 800,
    "y2": 600
}

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.4   # YOLO confidence threshold
IOU_THRESHOLD = 0.45         # NMS IoU threshold
MIN_CONF_FOR_RUN = 0.6       # min confidence to allow RUN (fail-safe)

# Jacket colour detection (black)
BLACK_THRESHOLD = 60         # B,G,R below this -> black pixel
JACKET_TOP_REL = 0.3         # top of jacket region inside bbox (relative)
JACKET_BOTTOM_REL = 0.8      # bottom of jacket region
JACKET_BLACK_RATIO_THRESHOLD = 0.4  # >= 40% black pixels => black jacket

# Logging
LOG_FILE = "safety_log.csv"


# =========================
# UTILITY FUNCTIONS
# =========================

def init_log_file(path: str) -> None:
    """Create CSV log with header if not exists."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "robot_state",          # RUN / STOP
                "reason",               # no_person / unauthorized_person / low_confidence / authorized_persons
                "num_persons_in_zone",
                "max_confidence"
            ])


def log_event(path: str, robot_state: str, reason: str,
              num_persons: int, max_conf: float) -> None:
    """Log one event row to CSV."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, robot_state, reason, num_persons, f"{max_conf:.3f}"])


def does_box_intersect_zone(x1: int, y1: int, x2: int, y2: int, zone: dict) -> bool:
    """Return True if bounding box [x1,y1,x2,y2] intersects the safety zone rectangle."""
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
# MAIN PIPELINE
# =========================

def main() -> None:
    # 1) Init log
    init_log_file(LOG_FILE)

    # 2) Load YOLO model
    print(f"[INFO] Loading YOLOv8 model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # 3) Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    robot_state = "RUN"
    cv2.namedWindow("Robot Cell Safety", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / no frame.")
            break

        start = time.time()
        persons_in_zone = []
        max_conf = 0.0

        # --- YOLO inference ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # We only care about "person" class
                if result.names[cls_id].lower() != "person":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if conf > max_conf:
                    max_conf = conf

                in_zone = does_box_intersect_zone(x1, y1, x2, y2, SAFETY_ZONE)

                # Draw bbox
                color = (0, 255, 0) if in_zone else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"person {conf:.2f}",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if in_zone:
                    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                    black_ratio = compute_jacket_black_ratio(crop)
                    authorized = is_authorized_jacket(black_ratio)

                    persons_in_zone.append({
                        "conf": conf,
                        "black_ratio": black_ratio,
                        "authorized": authorized
                    })

                    # Draw jacket region visualization
                    h = y2 - y1
                    j_top = int(y1 + JACKET_TOP_REL * h)
                    j_bottom = int(y1 + JACKET_BOTTOM_REL * h)
                    jacket_color = (255, 0, 0) if authorized else (0, 0, 255)
                    cv2.rectangle(frame, (x1, j_top), (x2, j_bottom), jacket_color, 1)
                    cv2.putText(frame, f"black={black_ratio:.2f}",
                                (x1, j_bottom + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # --- Safety decision (fail-safe) ---
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

        # Log state change
        if new_state != robot_state:
            robot_state = new_state
            log_event(LOG_FILE, robot_state, reason, len(persons_in_zone), max_conf)
            print(f"[EVENT] Robot={robot_state}, reason={reason}, "
                  f"persons_in_zone={len(persons_in_zone)}, max_conf={max_conf:.2f}")

        # Draw safety zone
        cv2.rectangle(frame,
                      (SAFETY_ZONE["x1"], SAFETY_ZONE["y1"]),
                      (SAFETY_ZONE["x2"], SAFETY_ZONE["y2"]),
                      (0, 0, 255), 2)

        # State text
        state_color = (0, 255, 0) if robot_state == "RUN" else (0, 0, 255)
        cv2.putText(frame, f"Robot: {robot_state}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    state_color,
                    2,
                    cv2.LINE_AA)

        # FPS
        dt = time.time() - start
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Robot Cell Safety", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished. Log saved to:", LOG_FILE)


if __name__ == "__main__":
    main()