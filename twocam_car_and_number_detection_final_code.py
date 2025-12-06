import os, time, re, math, csv
from datetime import datetime
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import serial  # <- for ESP32 serial connection

# ----------------- CONFIG -----------------
MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"
USE_GPU    = torch.cuda.is_available()
DEVICE     = 0 if USE_GPU else "cpu"

# Which camera is which (adjust indices to your PC)
CAM_ENTRY  = 1   # external (at the gate IN)
CAM_EXIT   = 0   # internal (at the gate OUT)

IMGSZ      = 640
CONF_DET   = 0.50

OCR_LANGS  = ["en"]
OCR_ALLOW  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
OCR_GPU    = USE_GPU

STABLE_FRAMES     = 4     # require N consecutive frames with a car before OCR
COOLDOWN_S        = 1.2   # min seconds between OCR attempts on the same cam
ROI_BOTTOM_RATIO  = 0.35  # use bottom part of box as plate region

# Light load tuning (skip frames so CPU survives two cameras)
STRIDE_ENTRY = 2  # process every 2nd frame on entry cam
STRIDE_EXIT  = 2  # process every 2nd frame on exit cam

# --------- BILLING ----------
BLOCK_SEC    = 60          # 1 minute
BLOCK_RATE   = 500         # kyats per minute
# ----------------------------

# ---------- LOGGING ----------
LOG_PATH = "parking_log.csv"
# ----------------------------

# ---------- ESP32 SERIAL CONFIG ----------
ARD_PORT = "COM7"     # <<< CHANGE THIS to your ESP32 COM port (e.g. "COM3", "COM7")
ARD_BAUD = 115200
# -----------------------------------------

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # faster open on Windows DSHOW
PLATE_RE = re.compile(r"^([0-9])([A-Z])\s?([0-9]{4})$")


# ---------- CSV logging helpers ----------
def _init_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "event", "camera", "plate",
                "start_time", "end_time", "duration_sec",
                "duration_hms", "blocks", "cost_kyats"
            ])


def log_event(event, camera, plate, start_ts=None, end_ts=None,
              duration_sec=None, blocks=None, cost=None):
    _init_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S") if start_ts else ""
    end_str   = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")   if end_ts   else ""
    dur = int(duration_sec) if duration_sec else 0
    dur_hms = f"{dur//3600:02d}:{(dur%3600)//60:02d}:{dur%60:02d}" if duration_sec else ""
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            ts, event, camera, plate,
            start_str, end_str,
            dur if duration_sec else "",
            dur_hms if duration_sec else "",
            blocks if blocks is not None else "",
            cost if cost is not None else ""
        ])


# ---------- Vision helpers ----------
def open_camera(idx: int, timeout_s: int = 6) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    start = time.time()
    while time.time() - start < timeout_s:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
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
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8
    )
    th = cv2.medianBlur(th, 3)
    return th


def clean_text(txt: str) -> str:
    txt = "".join(ch for ch in txt.upper() if ch in OCR_ALLOW)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def normalize_plate(raw: str) -> Optional[str]:
    s = clean_text(raw).replace(" ", "")
    if len(s) < 6:
        return None
    s = s[:6]

    to_digit  = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "J": "1",
                 "S": "5", "B": "8", "Z": "2", "G": "6"}
    to_letter = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}

    ch = list(s)
    if not ch[0].isdigit():
        ch[0] = to_digit.get(ch[0], ch[0])
    if ch[1].isdigit():
        ch[1] = to_letter.get(ch[1], ch[1])
    for i in range(2, 6):
        if not ch[i].isdigit():
            ch[i] = to_digit.get(ch[i], ch[i])
    s_fixed = "".join(ch)

    m = PLATE_RE.match(s_fixed) or PLATE_RE.match(f"{s_fixed[:2]} {s_fixed[2:]}")
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)} {m.group(3)}"


def best_plate_from_texts(texts: list[str]) -> Optional[str]:
    cand: list[str] = []
    for t in texts:
        t = clean_text(t)
        if t:
            cand += [t, t.replace(" ", "")]
    if texts:
        j = clean_text("".join(texts))
        if j:
            cand += [j, j.replace(" ", "")]
    for c in cand:
        p = normalize_plate(c)
        if p:
            return p
    return None


def detect_plate(yolo: YOLO,
                 reader: easyocr.Reader,
                 frame,
                 seen_streak: int,
                 last_ocr_time: float):
    """
    Return (annotated_frame, new_seen_streak, plate or None, new_last_ocr_time)
    """
    det = yolo.predict(
        source=frame,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF_DET,
        verbose=False
    )[0]
    annotated = det.plot()

    car_boxes = []
    if det.boxes is not None and len(det.boxes) > 0:
        for b in det.boxes:
            conf = float(b.conf.cpu().numpy()[0])
            if conf < CONF_DET:
                continue
            xyxy = b.xyxy.cpu().numpy()[0].astype(int)
            car_boxes.append((xyxy, conf))

    seen_streak = seen_streak + 1 if car_boxes else 0

    plate: Optional[str] = None
    now = time.time()

    if car_boxes and seen_streak >= STABLE_FRAMES and (now - last_ocr_time) > COOLDOWN_S:
        # take largest box
        xyxy, _ = max(
            car_boxes,
            key=lambda bc: (bc[0][2] - bc[0][0]) * (bc[0][3] - bc[0][1])
        )
        x1, y1, x2, y2 = xyxy
        h = y2 - y1
        roi_y1 = y1 + int((1.0 - ROI_BOTTOM_RATIO) * h)
        roi = frame[max(0, roi_y1):y2, max(0, x1):x2].copy()
        if roi.size > 0:
            prep = preprocess_for_ocr(roi)
            inv  = cv2.bitwise_not(prep)
            texts: list[str] = []
            texts += reader.readtext(
                prep, detail=0, paragraph=True, allowlist=OCR_ALLOW
            )
            texts += reader.readtext(
                inv,  detail=0, paragraph=True, allowlist=OCR_ALLOW
            )
            plate = best_plate_from_texts(texts)
        last_ocr_time = now

    return annotated, seen_streak, plate, last_ocr_time


# ---------- Billing helpers ----------
def ceil_minutes(seconds: float) -> int:
    return max(1, math.ceil(max(0.0, seconds) / BLOCK_SEC))


def fmt_hms(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


# ---------- MAIN ----------
def main():
    print("CUDA:", USE_GPU, "| DEVICE:", DEVICE)
    yolo = YOLO(MODEL_PATH)
    reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)

    # ---------- Arduino / ESP32 serial ----------
    print("Opening serial to ESP32...")
    ard = None
    try:
        ard = serial.Serial(ARD_PORT, ARD_BAUD, timeout=1)
        time.sleep(2)  # wait for ESP32 reset
        print(f"✅ Serial open on {ARD_PORT} @ {ARD_BAUD}")
    except Exception as e:
        print("❌ Could not open serial port:", e)
        print("   Entry/Exit gates will NOT move, but ANPR will run.")

    def send_open(which: str):
        """Send ENTRY / EXIT / CLOSE to ESP32 over serial."""
        if ard is None or not ard.is_open:
            print("⚠️ ESP32 serial not available, skipping gate command:", which)
            return
        if which == "ENTRY":
            msg = "ENTRY\n"
        elif which == "EXIT":
            msg = "EXIT\n"
        else:
            msg = "CLOSE\n"
        try:
            ard.write(msg.encode("ascii"))
            print("➡ Sent to ESP32:", msg.strip())
        except Exception as e2:
            print("❌ Serial write error:", e2)

    # ---------- Cameras ----------
    cap_in  = open_camera(CAM_ENTRY)  # ENTRY camera
    cap_out = open_camera(CAM_EXIT)   # EXIT camera

    cv2.namedWindow("ENTRY", cv2.WINDOW_NORMAL)
    cv2.namedWindow("EXIT",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ENTRY", 640, 360)
    cv2.resizeWindow("EXIT",  640, 360)

    # Keep simple session store: plate -> start_time
    active: dict[str, float] = {}

    # book-keeping per camera
    seen_in = seen_out = 0
    t_in = t_out = 0.0
    last_plate_in  = ""
    last_plate_out = ""

    i = j = 0
    print("✅ Press 'Q' in a window to quit")

    try:
        while True:
            ok_in,  frame_in  = cap_in.read()
            ok_out, frame_out = cap_out.read()
            if not ok_in and not ok_out:
                time.sleep(0.01)
                continue

            # ENTRY processing (stride to reduce load)
            if ok_in:
                i += 1
                if i % STRIDE_ENTRY == 0:
                    ann_in, seen_in, plate_in, t_in = detect_plate(
                        yolo, reader, frame_in, seen_in, t_in
                    )
                    if plate_in:
                        last_plate_in = plate_in
                        # Start session if not already active
                        if plate_in not in active:
                            active[plate_in] = time.time()
                            print(f"🅿 START | {plate_in} at {datetime.now().strftime('%H:%M:%S')}")
                            log_event("START", "ENTRY", plate_in,
                                      start_ts=active[plate_in])
                            # 🔴 OPEN ENTRY GATE
                            send_open("ENTRY")
                    else:
                        # if no plate, keep last ann_in
                        ann_in = ann_in
                else:
                    ann_in = frame_in

                if last_plate_in:
                    cv2.rectangle(ann_in, (10, 10), (360, 60), (0, 0, 0), -1)
                    cv2.putText(
                        ann_in, f"ENTRY: {last_plate_in}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA
                    )
                cv2.imshow("ENTRY", ann_in)

            # EXIT processing
            if ok_out:
                j += 1
                if j % STRIDE_EXIT == 0:
                    ann_out, seen_out, plate_out, t_out = detect_plate(
                        yolo, reader, frame_out, seen_out, t_out
                    )
                    if plate_out:
                        last_plate_out = plate_out
                        # If car is active, bill on exit
                        if plate_out in active:
                            start = active.pop(plate_out)
                            now   = time.time()
                            duration = now - start
                            blocks   = ceil_minutes(duration)
                            cost     = blocks * BLOCK_RATE
                            print(f"🧾 EXIT | {plate_out} | {fmt_hms(duration)} | {blocks}× | {cost} kyats")
                            log_event(
                                "RECEIPT", "EXIT", plate_out,
                                start_ts=start, end_ts=now,
                                duration_sec=duration, blocks=blocks,
                                cost=cost
                            )
                            # 🔴 OPEN EXIT GATE
                            send_open("EXIT")
                    else:
                        ann_out = ann_out
                else:
                    ann_out = frame_out

                if last_plate_out:
                    cv2.rectangle(ann_out, (10, 10), (360, 60), (0, 0, 0), -1)
                    cv2.putText(
                        ann_out, f"EXIT:  {last_plate_out}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA
                    )
                cv2.imshow("EXIT", ann_out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap_in.release()
        cap_out.release()
        cv2.destroyAllWindows()
        if ard is not None and ard.is_open:
            ard.close()
        print("🟢 Closed.")
        print(f"📄 Log saved to: {os.path.abspath(LOG_PATH)}")


if __name__ == "__main__":
    main()
