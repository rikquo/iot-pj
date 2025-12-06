# ocr_test_images.py
from ultralytics import YOLO
import easyocr
import cv2
import os
import re
import numpy as np

MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"  # your trained car detector
TEST_DIR = "test_images"                           # put sample images here
CONF_DET = 0.50                                    # car detection confidence
ROI_BOTTOM_RATIO = 0.35                            # bottom % of car box to OCR
OCR_ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "  # allowed characters

def clean_text(s: str) -> str:
    s = s.upper()
    s = re.sub(rf"[^{re.escape(OCR_ALLOW)}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 8)
    th = cv2.medianBlur(th, 3)
    return th

def main():
    yolo = YOLO(MODEL_PATH)
    reader = easyocr.Reader(["en"], gpu=True)

    if not os.path.isdir(TEST_DIR):
        raise SystemExit(f"Folder '{TEST_DIR}' not found. Create it and add test images.")

    for fn in os.listdir(TEST_DIR):
        p = os.path.join(TEST_DIR, fn)
        if not os.path.isfile(p):
            continue

        img = cv2.imread(p)
        if img is None:
            print(fn, "— could not read image"); continue

        r = yolo(img, conf=CONF_DET)[0]
        if r.boxes is None or len(r.boxes) == 0:
            print(fn, "— no car"); continue

        # take the largest detected car
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        x1,y1,x2,y2 = boxes[areas.argmax()]
        h = y2 - y1
        roi_y1 = y1 + int((1.0 - ROI_BOTTOM_RATIO) * h)
        roi = img[max(0, roi_y1):y2, max(0, x1):x2]

        if roi.size == 0:
            print(fn, "— empty ROI"); continue

        prep = preprocess_for_ocr(roi)
        txts = reader.readtext(prep, detail=0, paragraph=True, allowlist=OCR_ALLOW)

        if not txts:
            print(fn, "— no text"); continue

        candidate = clean_text(max(txts, key=len))
        print(fn, "→", candidate)

if __name__ == "__main__":
    main()
