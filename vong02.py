import cv2
import time
import math
import threading
import torch
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "bestbest.pt"      # Khuyến nghị: dùng .onnx / openvino / ncnn (xem hướng dẫn cuối file)
CAMERA_ID = 0
PROFILE = "low"                 # "normal" | "low" | "ultra_low"

SHOW_WINDOW = True              # False = chạy không hiển thị -> nhẹ hơn nữa
ADAPTIVE = True                 # Tự điều chỉnh số frame bỏ qua theo FPS
TARGET_FPS = 20                 # FPS mong muốn khi ADAPTIVE bật

PROFILES = {
    #               cam_w cam_h imgsz conf  detect_every max_det threads max_skip
    "normal":    dict(w=640, h=480, imgsz=640, conf=0.35, every=1, max_det=10, threads=4, max_skip=3),
    "low":       dict(w=320, h=240, imgsz=320, conf=0.40, every=2, max_det=5,  threads=2, max_skip=5),
    "ultra_low": dict(w=256, h=192, imgsz=256, conf=0.45, every=3, max_det=3,  threads=1, max_skip=8),
}

SMOOTHING = 0.35       # Càng nhỏ càng mượt
LOST_LIMIT = 4         # Số lần DETECT liên tiếp mất dấu trước khi tìm mục tiêu mới
# ==========================================

P = PROFILES[PROFILE]
CAM_W, CAM_H = P["w"], P["h"]
IMGSZ = P["imgsz"]
CONF = P["conf"]
MAX_DET = P["max_det"]
MAX_SKIP = P["max_skip"]
detect_every = P["every"]

# Giới hạn luồng CPU để không treo máy yếu
cv2.setNumThreads(P["threads"])
torch.set_num_threads(P["threads"])

use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else "cpu"
print("Device:", "CUDA GPU" if use_cuda else "CPU", "| Profile:", PROFILE)

model = YOLO(MODEL_PATH)


# ---------- Đọc camera ở thread riêng (luôn lấy frame mới nhất) ----------
class LatestFrameReader:
    def __init__(self, cam_id, w, h):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.frame = None
        self.new = False
        self.running = cap.isOpened()
        self.lock = threading.Lock()
        if self.running:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.running = False
                break
            with self.lock:
                self.frame = f
                self.new = True

    def read(self):
        """Trả về (frame, is_new)."""
        with self.lock:
            f, n = self.frame, self.new
            self.new = False
        return f, n

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def detect(frame):
    with torch.inference_mode():
        res = model.predict(
            frame,
            conf=CONF,
            imgsz=IMGSZ,
            device=device,
            half=use_cuda,
            max_det=MAX_DET,
            verbose=False,
        )
    boxes = res[0].boxes
    dets = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = b
            dets.append({
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "w": x2 - x1, "h": y2 - y1,
                "conf": float(c), "area": (x2 - x1) * (y2 - y1),
            })
    return dets


# ---------- Trạng thái theo dõi ----------
target = None          # dict: cx, cy, w, h, vx, vy, conf
lost = 0               # số lần detect liên tiếp mất dấu
since_det = 1          # số frame kể từ lần detect gần nhất
smooth_x = smooth_y = None
frame_idx = 0
force_reselect = False

fps = 0.0
prev_time = time.perf_counter()
adapt_counter = 0

reader = LatestFrameReader(CAMERA_ID, CAM_W, CAM_H)
if not reader.running:
    raise SystemExit("Không mở được camera")

while True:
    frame, is_new = reader.read()
    if frame is None or not is_new:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if not reader.running and frame is None:
            break
        time.sleep(0.002)
        continue

    h, w = frame.shape[:2]
    screen_x, screen_y = w // 2, h // 2
    frame_idx += 1

    # ----- Chạy YOLO mỗi N frame -----
    if frame_idx % detect_every == 0 or target is None:
        dets = detect(frame)

        if force_reselect:
            target = None
            force_reselect = False

        if target is not None:
            # Ghép mục tiêu gần nhất với vị trí dự đoán (trong vùng cho phép)
            gate = max(target["w"], target["h"]) * 1.5 + math.hypot(target["vx"], target["vy"]) * since_det
            best, best_d = None, 1e9
            for d in dets:
                dist = math.hypot(d["cx"] - target["cx"], d["cy"] - target["cy"])
                if dist < best_d:
                    best, best_d = d, dist
            if best is not None and best_d <= gate:
                n = max(since_det, 1)
                nvx = (best["cx"] - target["cx"]) / n
                nvy = (best["cy"] - target["cy"]) / n
                target["vx"] = 0.5 * target["vx"] + 0.5 * nvx
                target["vy"] = 0.5 * target["vy"] + 0.5 * nvy
                target.update(cx=best["cx"], cy=best["cy"], w=best["w"], h=best["h"], conf=best["conf"])
                lost = 0
            else:
                lost += 1
                if lost >= LOST_LIMIT:
                    target = None
                    smooth_x = smooth_y = None

        if target is None and dets:
            b = max(dets, key=lambda d: d["area"])
            target = dict(cx=b["cx"], cy=b["cy"], w=b["w"], h=b["h"],
                          vx=0.0, vy=0.0, conf=b["conf"])
            lost = 0
            smooth_x = smooth_y = None
        since_det = 1
    else:
        # ----- Frame bỏ qua: nội suy vị trí bằng vận tốc -----
        if target is not None:
            target["cx"] = min(max(target["cx"] + target["vx"], 0), w)
            target["cy"] = min(max(target["cy"] + target["vy"], 0), h)
            target["vx"] *= 0.92
            target["vy"] *= 0.92
        since_det += 1

    # ----- Vẽ (tối giản) -----
    if SHOW_WINDOW:
        cv2.circle(frame, (screen_x, screen_y), 4, (255, 255, 0), -1)

    if target is not None:
        cx, cy = target["cx"], target["cy"]
        if smooth_x is None:
            smooth_x, smooth_y = cx, cy
        else:
            smooth_x += SMOOTHING * (cx - smooth_x)
            smooth_y += SMOOTHING * (cy - smooth_y)

        obj_x, obj_y = int(smooth_x), int(smooth_y)
        dx, dy = obj_x - screen_x, obj_y - screen_y
        distance = math.hypot(dx, dy)

        if SHOW_WINDOW:
            x1 = int(cx - target["w"] / 2); y1 = int(cy - target["h"] / 2)
            x2 = int(cx + target["w"] / 2); y2 = int(cy + target["h"] / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(frame, (obj_x, obj_y), 4, (0, 0, 255), -1)
            cv2.arrowedLine(frame, (screen_x, screen_y), (obj_x, obj_y),
                            (0, 255, 255), 1, tipLength=0.05)
            cv2.putText(frame, f"D:{distance:.0f}px DX:{dx} DY:{dy}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Conf:{target['conf']:.2f}", (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        # Nếu cần điều khiển servo/motor: dùng dx, dy ở đây
    elif SHOW_WINDOW:
        cv2.putText(frame, "SEARCHING...", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # ----- FPS + tự điều chỉnh -----
    now = time.perf_counter()
    inst = 1.0 / max(now - prev_time, 1e-6)
    fps = 0.9 * fps + 0.1 * inst if fps else inst
    prev_time = now

    if ADAPTIVE:
        adapt_counter += 1
        if adapt_counter >= 30:
            adapt_counter = 0
            if fps < TARGET_FPS and detect_every < MAX_SKIP:
                detect_every += 1
            elif fps > TARGET_FPS * 1.6 and detect_every > 1:
                detect_every -= 1

    if SHOW_WINDOW:
        cv2.putText(frame, f"FPS:{fps:.0f} N:{detect_every}", (w - 110, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("YOLO TRACKING (LOW SPEC)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):              # Chọn lại vật thể
        force_reselect = True
        smooth_x = smooth_y = None

reader.release()
cv2.destroyAllWindows()

# =====================================================================
# HƯỚNG DẪN TĂNG TỐC THÊM (chạy 1 lần, rồi đổi MODEL_PATH sang file mới):
#
#   from ultralytics import YOLO
#   YOLO("bestbest.pt").export(format="onnx", imgsz=320, simplify=True)
#   # CPU Intel : format="openvino", imgsz=320, half=False
#   # ARM / Raspberry Pi : format="ncnn", imgsz=320
#
# Nếu model đang là yolov8m/l thì nên train lại bằng yolov8n (nano)
# hoặc yolo11n để nhanh gấp 3-5 lần trên CPU.
# =====================================================================