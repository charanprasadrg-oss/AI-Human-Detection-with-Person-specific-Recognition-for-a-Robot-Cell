This project implements a vision-based, edge-AI safety layer for industrial robot cells. A camera observes the robot workspace, and an embedded system (e.g., NVIDIA Jetson Nano) runs YOLOv8-based human detection plus additional logic to distinguish authorized vs. unauthorized persons and trigger safe robot behavior.

Conventional safety devices (light curtains, scanners) cannot distinguish humans from objects, causing false stops when robots, grippers, or workpieces enter the safety area. This system adds an intelligent human-only filter and person-specific logic on top of existing safety.

A YOLOv8 model (trained and managed in Roboflow) detects only the person class in each frame. Dataset images are extracted from robot-cell/factory videos; only humans are annotated, robots and workpieces are ignored. Roboflow is used for annotation, augmentation (resize, flip, brightness/contrast), versioning, and cloud training. The trained model is exported as a YOLOv8 .pt file and loaded by the Python code; for quick testing, the default yolov8n.pt can be used.

The main script (OpenCV + ultralytics + NumPy) captures frames from a webcam or CSI camera, runs YOLOv8 inference, and filters detections to persons inside a configurable rectangular safety zone. For each person in the zone, a “jacket region” slice of the bounding box is analyzed: the ratio of black pixels is computed to decide if the person is authorized (black jacket) or unauthorized (any other colour). Thresholds for safety zone, confidence, and jacket-black ratio are configurable.

A fail‑safe RUN/STOP state machine uses this information:

No person in zone → Robot = RUN.
Person in zone, non‑black jacket → Robot = STOP.
Low detection confidence → treated as unsafe → Robot = STOP.
Only high-confidence, authorized persons → Robot = RUN.
Every state change is logged to an Excel‑compatible CSV (safety_log.csv) with timestamp, state, reason, number of persons in zone, and maximum confidence. The visualization overlays bounding boxes, jacket regions, safety zone, robot state, and FPS. The system is designed for deployment on Jetson Nano with a Raspberry Pi camera, but can also run on a standard PC with a webcam or video file.
