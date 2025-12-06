# AI-Inspired Smart Parking Gate

**Description**
Smart parking gate with vehicle detection (YOLOv8), license plate OCR (EasyOCR), and billing logic. Controls gates via ESP32 serial commands.

## Quick setup
1. Configure `main` script variables:
   - `MODEL_PATH` (path to YOLO `.pt`)
   - `ARD_PORT` (ESP32 COM port, e.g. "COM7")
   - `CAM_ENTRY` and `CAM_EXIT` (camera indices)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run: `python detect_car_and_read_number.py`

## Hardware
- ESP32 or ESP32-CAM
- Two USB cameras (ENTRY, EXIT)
- Servo/gate mechanism controlled by ESP32

## Billing
- Block length: 60 seconds
- Price: 500 Kyats per block

## Files
- `detect_car_and_read_number.py` — main script
- `runs/car_yolo_gpu/weights/best.pt` — YOLO model
- `parking_log.csv` — event log
