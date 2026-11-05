
import cv2
import time
import math
import torch
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "bestbest.pt"
CAMERA_ID = 0

CONF = 0.35
IMGSZ = 640

SMOOTHING = 0.35       # Càng nhỏ càng mượt
LOST_LIMIT = 15        # Số frame mất dấu trước khi tìm mục tiêu mới

# ==========================================

device = 0 if torch.cuda.is_available() else "cpu"
use_cuda = torch.cuda.is_available()

print("Device:", "CUDA GPU" if use_cuda else "CPU")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

target_id = None
lost_frames = 0
smooth_x = None
smooth_y = None

fps = 0
prev_time = time.perf_counter()

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Tâm màn hình
    screen_x = w // 2
    screen_y = h // 2

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF,
        imgsz=IMGSZ,
        device=device,
        half=use_cuda,
        verbose=False
    )

    boxes = results[0].boxes
    detections = []

    if boxes is not None and len(boxes) > 0:

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        ids = None
        if boxes.id is not None:
            ids = boxes.id.int().cpu().numpy()

        for i, box in enumerate(xyxy):

            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tid = int(ids[i]) if ids is not None else i

            detections.append({
                "id": tid,
                "box": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
                "conf": float(confs[i]),
                "area": (x2-x1) * (y2-y1)
            })

    # Chọn mục tiêu và khóa ID
    selected = None

    if target_id is not None:
        for obj in detections:
            if obj["id"] == target_id:
                selected = obj
                break

    if selected is None:

        lost_frames += 1

        if lost_frames >= LOST_LIMIT:

            target_id = None
            smooth_x = None
            smooth_y = None

            if detections:
                selected = max(
                    detections,
                    key=lambda obj: obj["area"]
                )

                target_id = selected["id"]
                lost_frames = 0

    else:
        lost_frames = 0

    # Tâm màn hình
    cv2.circle(
        frame,
        (screen_x, screen_y),
        5,
        (255, 255, 0),
        -1
    )

    # Vẽ mục tiêu
    if selected is not None:

        x1, y1, x2, y2 = selected["box"]

        # Làm mượt tọa độ tâm vật thể
        current_x = selected["cx"]
        current_y = selected["cy"]

        if smooth_x is None:
            smooth_x = float(current_x)
            smooth_y = float(current_y)
        else:
            smooth_x += SMOOTHING * (current_x - smooth_x)
            smooth_y += SMOOTHING * (current_y - smooth_y)

        obj_x = int(smooth_x)
        obj_y = int(smooth_y)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Tâm vật thể
        cv2.circle(
            frame,
            (obj_x, obj_y),
            6,
            (0, 0, 255),
            -1
        )

        # Mũi tên từ tâm màn hình tới vật thể
        cv2.arrowedLine(
            frame,
            (screen_x, screen_y),
            (obj_x, obj_y),
            (0, 255, 255),
            2,
            tipLength=0.035
        )

        # Khoảng cách pixel
        dx = obj_x - screen_x
        dy = obj_y - screen_y

        distance = math.sqrt(dx**2 + dy**2)

        cv2.putText(
            frame,
            f"Distance: {distance:.1f} px",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ID: {target_id} | Conf: {selected['conf']:.2f}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"DX: {dx}  DY: {dy}",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "SEARCHING TARGET...",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

    # FPS
    current_time = time.perf_counter()
    instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
    fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("REALTIME YOLO OBJECT TRACKING", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Nhấn R để chọn lại vật thể
    if key == ord("r"):
        target_id = None
        lost_frames = LOST_LIMIT
        smooth_x = None
        smooth_y = None

cap.release()
cv2.destroyAllWindows()
