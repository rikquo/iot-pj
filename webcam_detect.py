from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"
CONF = 0.5
IMGSZ = 640
GPU = 0
CAM_CANDIDATES = [0]

def open_camera():
    for idx in CAM_CANDIDATES:
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release(); continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, _ = cap.read()
            if ok:
                print(f"✅ Opened cam index {idx} with backend {('MSMF' if backend==cv2.CAP_MSMF else 'DSHOW')}")
                return cap
            cap.release()
    raise SystemExit("❌ No webcam opened. Adjust CAM_CANDIDATES indices.")

def main():
    model = YOLO(MODEL_PATH)
    cap = open_camera()

    window_name = "Hot Wheels Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    stride = 1
    i = 0
    last_ok = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if time.time() - last_ok > 2.0:
                    print("⚠️ No frames for 2s, exiting cleanly.")
                    break
                continue
            last_ok = time.time()
            i += 1
            if i % stride != 0:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 🔽 darken / restore to normal brightness before prediction
            frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=-17)
            # alpha < 1 lowers contrast slightly, beta < 0 lowers brightness
            # try alpha=0.85,beta=-35 if still too bright

            results = model.predict(
                source=frame,
                device=GPU,
                conf=CONF,
                imgsz=IMGSZ,
                verbose=False
            )

            annotated = results[0].plot()
            cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🟢 Webcam closed cleanly.")

if __name__ == "__main__":
    main()
