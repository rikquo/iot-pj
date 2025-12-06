import os
import time
import re
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# ----------------- CONFIG -----------------
MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"  # your trained car detector
CAM_INDEX = 1                                     # 1 if your external cam is index 1
DEVICE = 0                                        # 0 -> GPU for YOLO, or "cpu"
IMGSZ = 640                                       # YOLO inference size
CONF_DET = 0.50                                   # car detection confidence

OCR_GPU = True                                    # use GPU for EasyOCR if available
OCR_LANGS = ["en"]                                # OCR language(s)
OCR_ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "  # restrict chars

STABLE_FRAMES = 5                                 # require car present N consecutive frames
COOLDOWN_S = 1.5                                  # time between OCR runs on same car
ROI_BOTTOM_RATIO = 0.35                           # take bottom 35% of car box as plate/number area
# ------------------------------------------

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # prefer DSHOW on Windows for fast open

# strict target pattern: 1 digit + 1 letter + optional space + 4 digits
PLATE_RE = re.compile(r"^([0-9])([A-Z])\s?([0-9]{4})$")


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
    """Light denoise + contrast; adaptive threshold for ink-on-paper signs."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    th = cv2.medianBlur(th, 3)
    return th


def clean_text(txt: str) -> str:
    txt = "".join(ch for ch in txt.upper() if ch in OCR_ALLOW)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def normalize_plate(raw: str) -> str | None:
    """
    Try to coerce OCR text into the strict plate format:
      digit + letter + [space] + 4 digits
    Fixes common OCR confusions in the correct positions.
    Returns formatted 'X L DDDD' or None if invalid.
    """
    s = clean_text(raw).replace(" ", "")  # remove spaces first

    if len(s) < 6:
        return None

    # take first 6 plausible characters
    s = s[:6]

    # maps for positions that must be digits
    to_digit = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "J": "1",
        "S": "5", "B": "8", "Z": "2", "G": "6"
    }
    # map for the position that must be a letter (second char)
    to_letter = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}

    chars = list(s)

    # pos 0 -> must be digit
    if not chars[0].isdigit():
        chars[0] = to_digit.get(chars[0], chars[0])

    # pos 1 -> must be letter
    if chars[1].isdigit():
        chars[1] = to_letter.get(chars[1], chars[1])

    # last 4 -> must be digits
    for i in range(2, 6):
        if not chars[i].isdigit():
            chars[i] = to_digit.get(chars[i], chars[i])

    s_fixed = "".join(chars)

    # Validate final pattern (with or without space)
    m = PLATE_RE.match(s_fixed) or PLATE_RE.match(f"{s_fixed[:2]} {s_fixed[2:]}")
    if not m:
        return None

    # Return pretty format with a space
    return f"{m.group(1)}{m.group(2)} {m.group(3)}"


def best_plate_from_texts(texts: list[str]) -> str | None:
    """
    Try multiple candidates: each text, joined text, and space-stripped text.
    Return the first valid normalized plate, or None.
    """
    candidates = []
    for t in texts:
        t = clean_text(t)
        if t:
            candidates.append(t)
            candidates.append(t.replace(" ", ""))

    if texts:
        joined = clean_text("".join(texts))
        if joined:
            candidates.append(joined)
            candidates.append(joined.replace(" ", ""))

    for c in candidates:
        plate = normalize_plate(c)
        if plate:
            return plate
    return None


def main():
    print("Loading YOLO…")
    yolo = YOLO(MODEL_PATH)

    print("Loading EasyOCR…")
    reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)

    cap = open_camera(CAM_INDEX)
    win = "Car + OCR"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    seen_streak = 0
    last_ocr_time = 0.0
    last_read = ""

    print("✅ Press 'Q' to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # 1) Car detection
            det = yolo.predict(
                source=frame, device=DEVICE, imgsz=IMGSZ, conf=CONF_DET, verbose=False
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

            # 2) Run OCR only if car is stable for a few frames and cooldown passed
            now = time.time()
            if car_boxes and seen_streak >= STABLE_FRAMES and (now - last_ocr_time) > COOLDOWN_S:
                # largest car box
                xyxy, _ = max(
                    car_boxes,
                    key=lambda bc: (bc[0][2]-bc[0][0]) * (bc[0][3]-bc[0][1])
                )
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1

                # crop bottom portion where paper number usually is
                roi_y1 = y1 + int((1.0 - ROI_BOTTOM_RATIO) * h)
                roi = frame[max(0, roi_y1):y2, max(0, x1):x2].copy()

                if roi.size > 0:
                    prep = preprocess_for_ocr(roi)

                    # Try normal + inverted for robustness
                    inv = cv2.bitwise_not(prep)
                    texts = []
                    texts += reader.readtext(prep, detail=0, paragraph=True, allowlist=OCR_ALLOW)
                    texts += reader.readtext(inv,  detail=0, paragraph=True, allowlist=OCR_ALLOW)

                    plate = best_plate_from_texts(texts)
                    if plate:
                        last_read = plate
                        last_ocr_time = now
                        print(f"📖 OCR: {last_read}")

            # Overlay last OCR result
            if last_read:
                cv2.rectangle(annotated, (10, 10), (10+360, 60), (0, 0, 0), -1)
                cv2.putText(
                    annotated, f"NUM: {last_read}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA
                )

            cv2.imshow(win, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🟢 Closed.")


if __name__ == "__main__":
    main()
