"""PoC ตรวจจับลูกบอลจากไฟล์ MP4 แบบจำลอง real-time (v2 งานถ่ายทอดสด/ไฮไลต์)."""
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------- ตั้งค่า ----------
VIDEO_PATH = "dataset/1.mp4"  # path สัมพัทธ์จากโฟลเดอร์งาน
IMGSZ = 640           # ภาพถ่ายทอดสดบอลลูกเล็ก เบลอ 320 หายเกลี้ยง ต้อง 640
CONF = 0.15           # บอลเล็ก+เบลอ conf ต่ำ ลด miss ยอม false บ้าง
INFER_EVERY = 2       # ยิงทุก 2 เฟรม ชดเชยค่า imgsz ที่แพงขึ้น
MIN_SIZE = 8          # ภาพ wide บอลจิ๋วได้ถึง ~8px
MAX_SIZE = 150        # ภาพซูมใกล้บอลใหญ่เกิน 60px ได้
ASPECT_LO = 0.6       # บอลเบลอขณะเคลื่อนที่ยืดเป็นวงรีได้
ASPECT_HI = 1.4
USE_COLOR_FILTER = False  # ปิดฟิลเตอร์ขาวเป็นค่าเริ่ม บอลจริงมีลายดำ+เงา+เบลอ ตรวจขาวตกหมด
WHITE_THRESH = 0.15   # ใช้เฉพาะตอนเปิด USE_COLOR_FILTER
MAX_TRACK_MISSED = 10
MATCH_DIST = 100      # กล้องแพน/ซูม บอลกระโดดไกลระหว่างเฟรม
SCENE_CUT_THRESH = 40.0  # คลิปไฮไลต์ตัดช็อตบ่อย เจอตัดช็อตทิ้งแทร็กเก่า


def is_mostly_white(roi, thresh=WHITE_THRESH):
    """คืน True ถ้าพื้นที่ ROI มีพิกเซลสีขาวเกิน thresh."""
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
    return white_ratio > thresh


def pick_best_ball(frame, model, imgsz=IMGSZ, conf=CONF,
                   min_size=MIN_SIZE, max_size=MAX_SIZE,
                   use_color=USE_COLOR_FILTER, white_thresh=WHITE_THRESH):
    """ยิง YOLO เต็มเฟรมแล้วคืนกล่องบอลดีที่สุด 1 กล่อง (ไม่มีครอป ROI)."""
    results = model.predict(source=frame, imgsz=imgsz, conf=conf,
                            classes=[32], verbose=False)
    if not results or results[0].boxes is None:
        return None

    best, max_conf = None, 0.0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_v = float(box.conf[0])
        bw, bh = x2 - x1, y2 - y1

        if bw < min_size or bh < min_size or bw > max_size or bh > max_size:
            continue
        aspect = bw / float(bh)
        if not (ASPECT_LO <= aspect <= ASPECT_HI):
            continue
        if use_color:
            # เช็กเฉพาะแกนกลาง 60% ของกล่อง เลี่ยงขอบที่ปนพื้นหลัง
            ix1 = x1 + bw // 5
            ix2 = x2 - bw // 5
            iy1 = y1 + bh // 5
            iy2 = y2 - bh // 5
            roi = frame[max(0, iy1):iy2, max(0, ix1):ix2]
            if not is_mostly_white(roi, white_thresh):
                continue
        if conf_v > max_conf:
            max_conf = conf_v
            best = (x1, y1, x2, y2, conf_v)
    return best


def is_scene_cut(prev_thumb, frame, thresh=SCENE_CUT_THRESH):
    """เทียบภาพย่อขาวดำกับเฟรมก่อนหน้า ตัดช็อตใหม่คืน True."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
    if prev_thumb is None:
        return False, small
    diff = float(np.mean(cv2.absdiff(small, prev_thumb)))
    return diff > thresh, small


def main():
    video_file = Path(VIDEO_PATH)
    if not video_file.exists():
        raise FileNotFoundError(f"ไม่พบคลิป: {VIDEO_PATH}")

    model = YOLO("yolov8n.pt")  # รันครั้งแรกจะดาวน์โหลดน้ำหนักอัตโนมัติ
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError("เปิดวิดีโอไม่ได้ ตรวจ path หรือ codec")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(1000.0 / src_fps) - 5)
    print(f"src_fps={src_fps:.1f} infer_every={INFER_EVERY} imgsz={IMGSZ}")

    track = None
    missed = 0
    prev_thumb = None
    ema_fps = 0.0
    frame_id = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        do_infer = (frame_id % INFER_EVERY == 1)

        if do_infer:
            cut, prev_thumb = is_scene_cut(prev_thumb, frame)
            if cut:  # ตัดช็อตใหม่ กล่องเก่าใช้ไม่ได้ ทิ้งทันที
                track, missed = None, 0
            det = pick_best_ball(frame, model)
            if det is not None:
                track = det
                missed = 0
            else:
                missed += 1
                if missed > MAX_TRACK_MISSED:
                    track = None

        vis = frame.copy()
        if track is not None:
            x1, y1, x2, y2, conf_v = track
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = (0, 0, 255) if do_infer else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(vis, f"Ball {conf_v:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        inst_fps = 1.0 / max(1e-6, time.perf_counter() - t0)
        ema_fps = inst_fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst_fps
        cv2.putText(vis, f"FPS {ema_fps:.1f} f{frame_id}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Ball PoC v2 (q=quit)", vis)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"จบ {frame_id} เฟรม avg_fps~{ema_fps:.1f}")


if __name__ == "__main__":
    main()
