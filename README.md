# AI-Inspired Smart Parking Gate

**Description**
Smart parking gate with vehicle detection (YOLOv8), license plate OCR (EasyOCR), and billing logic. Controls gates via ESP32 serial commands.

⚠️ Note: yolov8n.pt (~6 MB) is included for convenience.
If cloning is slow, you can delete it and download YOLO models manually:
https://github.com/ultralytics/assets

## Automatic Vehicle Detection • License Plate OCR • Real-Time Billing • ESP32 Gate Control

A complete IoT-based smart parking system that uses YOLOv8 for car detection, EasyOCR for plate recognition, and ESP32 for controlling the entry/exit gate barrier.
The system logs entry/exit events, calculates billing, and automates gate movements based on license plate verification.

🔒 License Notice
This project is provided for educational and personal use only.
Commercial use, resale, or redistribution is strictly prohibited.

## Features
- Vehicle Detection (YOLOv8)
Detects incoming and outgoing vehicles using two webcams.

-License Plate Recognition (EasyOCR)
Extracts and normalizes plate numbers (e.g., "7N 4532").

-Automated Entry/Exit Gates (ESP32)
Sends serial commands to open/close gate servos.

-Real-Time Billing System
Billing is calculated based on time spent inside the parking area.

-Logging System
All events saved to parking_log.csv:

Entry time
Exit time
Duration
Billing amount
Plate number

## System Workflow (How It Works)

Below is a clear step-by-step breakdown of the entire system:

### 1. Startup

Loads YOLO model (best.pt)

Starts EasyOCR

Opens serial communication with ESP32

Initializes ENTRY + EXIT cameras

### 2. Entry Gate Process

ENTRY camera detects a car using YOLO

The bottom ROI of the car is scanned for the license plate

EasyOCR extracts plate text

Plate is cleaned → normalized → validated

If a valid plate is detected:

System creates an entry session

Saves entry record to parking_log.csv

Sends command ENTRY → ESP32 opens the gate

### 3. Exit Gate Process

EXIT camera detects car

OCR reads the plate

System checks if plate exists in active sessions

If yes:

Calculates total parking time

Computes billing

Logs receipt to CSV

Sends EXIT → Gate opens

### 4. Continuous Loop

The system runs non-stop until the user presses Q to quit.

## 🔌 Hardware Requirements

Microcontroller
ESP32 / ESP32-CAM
Servo motor (gate arm)
Ultrasonic sensor (optional)
RGB LED Module (optional)
Cameras
2x USB Webcams
ENTRY camer
EXIT camera

### Computer
Laptop/PC capable of running YOLO (GPU recommended)

## 🛠 Software Requirements

Python
Python 3.10 or 3.11

Dependencies
Install automatically:
pip install -r requirements.txt

Recommended IDE
VS Code / PyCharm

## Configuration
Edit top of the Python script (twocam_car_and_number_detection_final_code.py):

YOLO Model Path
MODEL_PATH = "runs/car_yolo_gpu/weights/best.pt"

ESP32 Serial Port
ARD_PORT = "COM7"

Camera Index
CAM_ENTRY = 1
CAM_EXIT = 0


Try switching numbers if the wrong camera opens.

Billing Settings
BLOCK_SEC  = 60
BLOCK_RATE = 500

## Billing Logic

1 block = 60 seconds
Rate = 500 Kyats per block
Always rounded UP (ceil)
Example:

Total Time: 130 sec
Blocks: 3
Billing: 3 × 500 = 1500 Kyats

## 📁 Project Structure

AI-Smart-Parking-Gate/
│
├── twocam_car_and_number_detection_final_code.py   # Main detection + OCR + billing + ESP32
├── requirements.txt                                # Dependency list
├── parking_log.csv                                 # Logs (auto-generated)
│
├── runs/                                           # YOLO model + outputs
│   └── car_yolo_gpu/weights/best.pt
│
├── test_images/                                    # For YOLO testing (optional)
├── ocr_test_images/                                # For OCR testing (optional)
├── webcam_detect/                                  # Stored webcam frames (optional)
│
├── data.yaml                                       # YOLO dataset config (if included)
└── esp32/                                          # ESP32 gate control code
    ├── entry_gate.ino
    └── exit_gate.ino

## Sample Output (Console)
[ENTRY] Car detected
[ENTRY] Plate: 7N 4532
[ENTRY] Session started
[ESP32] ENTRY gate opened

## Sample Log Entry (CSV)
timestamp,event,camera,plate,start_time,end_time,duration_sec,blocks,cost
2025-02-15 12:03:22,START,ENTRY,7N 4532,...
2025-02-15 12:10:55,RECEIPT,EXIT,7N 4532,...,453,8,4000

## Running the Program
python twocam_car_and_number_detection_final_code.py

Press Q anytime to stop the system.

## 🔒 License

This project is for educational, learning, and personal portfolio use only.
Commercial use, resale, or redistribution is NOT allowed without permission.


