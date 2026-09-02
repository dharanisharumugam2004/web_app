# HazWaste Detection

## Overview

**HazWaste Detection** is an AI-powered hazardous waste detection system that uses **YOLO object detection models** to identify hazardous waste objects from images, videos, and live webcam input.

The application provides a web-based interface where users can:

- Upload an image and detect hazardous waste.
- Upload a video and perform frame-by-frame detection.
- Use a webcam for real-time detection.
- Select different YOLO models for comparison.
- Adjust detection confidence and IoU thresholds.
- View detection statistics and inference results.
- Download or view processed detection results.

The backend is built using **FastAPI**, while the frontend uses **HTML, CSS, and JavaScript**.

---

## Features

### 1. Image Detection

Upload an image and run object detection using the selected YOLO model.

The system returns:

- Detected objects
- Bounding boxes
- Confidence scores
- Processed image
- Detection statistics
- Inference time

---

### 2. Video Detection

Upload a video and process it frame by frame.

The system performs detection on the video and generates a processed output containing bounding boxes around detected objects.

---

### 3. Real-Time Webcam Detection

The application supports webcam-based detection.

Frames captured from the webcam are sent to the FastAPI backend, where the selected YOLO model performs inference and returns the annotated frame.

---

### 4. Multiple YOLO Models

The application supports six YOLO model variants:

| Model | Type |
|---|---|
| YOLOv8-S | Small |
| YOLO11-S | Small |
| YOLO26-S | Small |
| YOLOv8-N | Nano |
| YOLO11-N | Nano |
| YOLO26-N | Nano |

The model can be selected directly from the web interface.

> **Note:** Each model must have its corresponding trained `best.pt` checkpoint available in the project directory.

---

## Supported Detection Models

The expected model directory structure is:

```text
models/
├── yolov8s/
│   └── best.pt
│
├── yolo11s/
│   └── best.pt
│
├── yolo26s/
│   └── best.pt
│
├── yolov8n/
│   └── best.pt
│
├── yolo11n/
│   └── best.pt
│
└── yolo26n/
    └── best.pt