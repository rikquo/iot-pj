# AI-Inspired Smart Parking Gate

**Description**
Smart parking gate with vehicle detection (YOLOv8), license plate OCR (EasyOCR), and billing logic. Controls gates via ESP32 serial commands.

## Setup
1. Configure `main` script variables:
   - `MODEL_PATH` (path to YOLO `.pt`)
   - `ARD_PORT` (ESP32 COM port, e.g. "COM7")
   - `CAM_ENTRY` and `CAM_EXIT` (camera indices)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run: `python detect_car_and_read_number.py`

## Hardware
- ESP32 or ESP32-CAM or External WebCam
- Two USB cameras (ENTRY, EXIT)
- Servo/gate mechanism controlled by ESP32

## Billing
- Block length: 60 seconds
- Price: 500 Kyats per block

## Files
- `twocam_car_and_number_detection_final_code.py` — main script
- `runs/car_yolo_gpu/weights/best.pt` — YOLO model
- `parking_log.csv` — event log

## 🔄 How the Smart Parking Gate Works (Step-by-Step)

### 1. System Initialization
- The Python script starts.
- YOLO model is loaded (`best.pt`).
- EasyOCR is initialized.
- The script attempts to connect to the ESP32 through serial communication.

### 2. Camera Feeds Start
- ENTRY camera and EXIT camera are opened.
- The system continuously reads frames from both cameras in real time.

### 3. Vehicle Detection (ENTRY)
- YOLO checks if a car is present at the entry gate.
- If detected:
  - OCR extracts the license plate number.
  - Plate is validated (must match allowed characters).
  - Entry time and car number plate is saved in `parking_log.csv`.
  - Python sends **OPEN_ENTRY** to the ESP32.
  - Gate opens → car enters → gate closes.

### 4. Vehicle Detection (EXIT)
- YOLO checks if a car is present at the exit gate.
- If detected:
  - OCR reads the plate.
  - Exit time is saved.
  - System retrieves matching entry time from CSV.
  - **Billing is calculated**:
    - Price = `(Total seconds / BLOCK_SEC) * BLOCK_RATE`
  - Full record saved to `parking_log.csv`.
  - Python sends **OPEN_EXIT** to ESP32.

### 5. Loop Continues
- System runs forever until manually stopped.

