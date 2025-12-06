import os
import time
import re
import math
import csv
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# ----------------- CONFIG -----------------
MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"
CAM_INDEX  = 1
DEVICE     = 0
IMGSZ      = 640
CONF_DET   = 0.50

OCR_GPU    = True
OCR_LANGS  = ["en"]
OCR_ALLOW  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

STABLE_FRAMES     = 5
COOLDOWN_S        = 1.5
ROI_BOTTOM_RATIO  = 0.35
# ------------------------------------------

# ------- BILLING (pay when the car RETURNS) -------
BLOCK_SEC    = 60          # 1 minute
BLOCK_RATE   = 500         # kyats per minute
GONE_TIMEOUT = 12          # seconds: consider car "gone" after this idle
# --------------------------------------------------

# ------- LOGGING -------
LOG_PATH = "parking_log.csv"   # Excel-friendly CSV
# -----------------------

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
PLATE_RE = re.compile(r"^([0-9])([A-Z])\s?([0-9]{4})$")

# ---------- CSV logging helpers ----------
def _init_log():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "event", "plate",
                "start_time", "end_time",
                "duration_sec", "duration_hms",
                "blocks", "cost_kyats"
            ])

def log_event(event: str, plate: str,
             start_ts: float | None = None,
             end_ts: float | None = None,
             duration_sec: float | None = None,
             blocks: int | None = None,
             cost: int | None = None):
    """Append one row to CSV."""
    _init_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S") if start_ts else ""
    end_str   = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")   if end_ts   else ""
    dur = int(duration_sec) if duration_sec else 0
    dur_hms = f"{dur//3600:02d}:{(dur%3600)//60:02d}:{dur%60:02d}" if duration_sec else ""
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            ts, event, plate,
            start_str, end_str,
            dur if duration_sec else "",
            dur_hms if duration_sec else "",
            blocks if blocks is not None else "",
            cost if cost is not None else ""
        ])

# ---------- OCR helpers ----------
def open_camera(idx: int, timeout_s: int = 6) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    start = time.time()
    while time.time() - start < timeout_s:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                print(f"✅ Camera {idx} ready in {time.time()-start:.2f}s")
                return cap
        time.sleep(0.1)
    cap.release()
    raise SystemExit(f"❌ Could not open camera index {idx}")

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 8)
    th = cv2.medianBlur(th, 3)
    return th

def clean_text(txt: str) -> str:
    txt = "".join(ch for ch in txt.upper() if ch in OCR_ALLOW)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def normalize_plate(raw: str) -> str | None:
    s = clean_text(raw).replace(" ", "")
    if len(s) < 6:
        return None
    s = s[:6]
    to_digit  = {"O":"0","Q":"0","D":"0","I":"1","L":"1","J":"1","S":"5","B":"8","Z":"2","G":"6"}
    to_letter = {"0":"O","1":"I","5":"S","2":"Z","8":"B"}
    ch = list(s)
    if not ch[0].isdigit(): ch[0] = to_digit.get(ch[0], ch[0])
    if ch[1].isdigit():     ch[1] = to_letter.get(ch[1], ch[1])
    for i in range(2,6):
        if not ch[i].isdigit(): ch[i] = to_digit.get(ch[i], ch[i])
    s_fixed = "".join(ch)
    m = PLATE_RE.match(s_fixed) or PLATE_RE.match(f"{s_fixed[:2]} {s_fixed[2:]}")
    if not m: return None
    return f"{m.group(1)}{m.group(2)} {m.group(3)}"

def best_plate_from_texts(texts: list[str]) -> str | None:
    candidates = []
    for t in texts:
        t = clean_text(t)
        if t:
            candidates += [t, t.replace(" ", "")]
    if texts:
        joined = clean_text("".join(texts))
        if joined:
            candidates += [joined, joined.replace(" ", "")]
    for c in candidates:
        p = normalize_plate(c)
        if p: return p
    return None

# ---------- Session (pay on return) ----------
class Session:
    __slots__ = ("plate","start_time","last_seen","present")
    def __init__(self, plate: str, t: float):
        self.plate = plate
        self.start_time = t
        self.last_seen  = t
        self.present    = True

def ceil_minutes(seconds: float) -> int:
    return max(1, math.ceil(max(0.0, seconds) / BLOCK_SEC))

def fmt_hms(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"

def main():
    print("Loading YOLO…")
    yolo = YOLO(MODEL_PATH)
    print("Loading EasyOCR…")
    reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)

    cap = open_camera(CAM_INDEX)
    win = "Smart Parking (Pay on Return)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    seen_streak   = 0
    last_ocr_time = 0.0
    last_read     = ""

    sessions: dict[str, Session] = {}

    print("✅ Press 'Q' to quit")
    try:
        while True:
            now = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            det = yolo.predict(source=frame, device=DEVICE, imgsz=IMGSZ,
                               conf=CONF_DET, verbose=False)[0]
            annotated = det.plot()

            # collect car boxes
            car_boxes = []
            if det.boxes is not None and len(det.boxes) > 0:
                for b in det.boxes:
                    conf = float(b.conf.cpu().numpy()[0])
                    if conf < CONF_DET: continue
                    xyxy = b.xyxy.cpu().numpy()[0].astype(int)
                    car_boxes.append((xyxy, conf))

            # mark sessions "absent" if not seen for GONE_TIMEOUT
            for plate, s in sessions.items():
                if s.present and (now - s.last_seen > GONE_TIMEOUT):
                    s.present = False

            seen_streak = seen_streak + 1 if car_boxes else 0

            # OCR only when stable + cooldown
            if car_boxes and seen_streak >= STABLE_FRAMES and (now - last_ocr_time) > COOLDOWN_S:
                xyxy, _ = max(car_boxes, key=lambda bc: (bc[0][2]-bc[0][0])*(bc[0][3]-bc[0][1]))
                x1, y1, x2, y2 = xyxy
                h = y2 - y1
                roi_y1 = y1 + int((1.0 - ROI_BOTTOM_RATIO) * h)
                roi = frame[max(0, roi_y1):y2, max(0, x1):x2].copy()

                if roi.size > 0:
                    prep = preprocess_for_ocr(roi)
                    inv  = cv2.bitwise_not(prep)
                    texts = []
                    texts += reader.readtext(prep, detail=0, paragraph=True, allowlist=OCR_ALLOW)
                    texts += reader.readtext(inv,  detail=0, paragraph=True, allowlist=OCR_ALLOW)

                    plate = best_plate_from_texts(texts)
                    last_ocr_time = now

                    if plate:
                        last_read = plate
                        s = sessions.get(plate)

                        if s is None:
                            # first ever sighting -> start a session
                            s = Session(plate, now)
                            sessions[plate] = s
                            print(f"🅿 START | {plate} at {datetime.now().strftime('%H:%M:%S')}")
                            log_event("START", plate, start_ts=s.start_time)
                        else:
                            if not s.present:
                                # car returned -> BILL NOW (from previous start to now)
                                duration = now - s.start_time
                                blocks   = ceil_minutes(duration)
                                cost     = blocks * BLOCK_RATE
                                print(f"🧾 Receipt | {plate} | parked {fmt_hms(duration)} "
                                      f"| {blocks}× blocks | {cost} kyats")
                                # log receipt
                                log_event(
                                    "RECEIPT",
                                    plate,
                                    start_ts=s.start_time,
                                    end_ts=now,
                                    duration_sec=duration,
                                    blocks=blocks,
                                    cost=cost
                                )
                                # reset start for next cycle
                                s.start_time = now
                                s.present    = True

                        # Update last seen timestamp
                        s.last_seen = now

            # overlays
            if last_read:
                cv2.rectangle(annotated, (10, 10), (430, 70), (0, 0, 0), -1)
                cv2.putText(annotated, f"NUM: {last_read}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"sessions: {len(sessions)}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow(win, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🟢 Closed.")
        print(f"📄 Log saved to: {os.path.abspath(LOG_PATH)}")

if __name__ == "__main__":
    main()
