import base64
import os
import time
import uuid
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="HazWaste Detection",
    description="Hazardous Waste Object Detection using YOLO",
    version="1.0.0"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI TEMPLATE + STATIC FILES
# ============================================================

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

MAX_CONTENT_LENGTH = 50 * 1024 * 1024

IMAGE_SIZE = 640

DEFAULT_CONFIDENCE = 0.40

DEFAULT_IOU = 0.50


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {

    # -------------------------
    # YOLOv8
    # -------------------------

    "YOLOv8-S": MODEL_DIR / "yolov8s" / "best.pt",

    "YOLOv8-N": MODEL_DIR / "yolov8n" / "best.pt",


    # -------------------------
    # YOLO11
    # -------------------------

    "YOLO11-S": MODEL_DIR / "yolo11s" / "best.pt",

    "YOLO11-N": MODEL_DIR / "yolo11n" / "best.pt",


    # -------------------------
    # YOLO26
    # -------------------------

    "YOLO26-S": MODEL_DIR / "yolo26s" / "best.pt",

    "YOLO26-N": MODEL_DIR / "yolo26n" / "best.pt",
}


# ============================================================
# DEFAULT MODEL
# ============================================================

DEFAULT_MODEL = "YOLO26-S"


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"


print("=" * 60)
print("HazWaste Detection")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available - using CPU")

print("=" * 60)


# ============================================================
# MODEL CACHE
# ============================================================

_loaded_model = None

_loaded_model_name = None

_model_lock = Lock()


# ============================================================
# RESULT STORAGE
# ============================================================

results = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_float(
    value,
    default,
    minimum=None,
    maximum=None
):
    """
    Safely convert a value to float.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name: str):

    global _loaded_model
    global _loaded_model_name

    # --------------------------------------------------------
    # Validate model name
    # --------------------------------------------------------

    if model_name not in MODEL_PATHS:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    model_path = MODEL_PATHS[model_name]


    # --------------------------------------------------------
    # Check model file
    # --------------------------------------------------------

    if not model_path.is_file():

        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )


    # --------------------------------------------------------
    # Lock model loading
    # --------------------------------------------------------

    with _model_lock:

        # If requested model is already loaded,
        # reuse it.

        if (
            _loaded_model is not None
            and _loaded_model_name == model_name
        ):

            return _loaded_model


        print()
        print("=" * 60)
        print(f"Loading model: {model_name}")
        print(f"Checkpoint: {model_path}")
        print("=" * 60)


        # ----------------------------------------------------
        # Load YOLO model
        # ----------------------------------------------------

        model = YOLO(
            str(model_path)
        )


        # ----------------------------------------------------
        # Move model to device
        # ----------------------------------------------------

        try:

            model.to(DEVICE)

        except Exception as e:

            print(
                f"Warning: could not explicitly move model "
                f"to device {DEVICE}: {e}"
            )


        # ----------------------------------------------------
        # Update cache
        # ----------------------------------------------------

        _loaded_model = model

        _loaded_model_name = model_name


        print(
            f"Model loaded successfully: {model_name}"
        )

        return model


# ============================================================
# DETECT FRAME
# ============================================================

def detect_frame(
    frame,
    model_name,
    confidence=DEFAULT_CONFIDENCE,
    iou=DEFAULT_IOU
):

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_name
    )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    start_time = time.perf_counter()


    prediction_results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=iou,
        device=DEVICE,
        verbose=False
    )


    inference_time = (
        time.perf_counter() - start_time
    )


    # --------------------------------------------------------
    # Get first result
    # --------------------------------------------------------

    result = prediction_results[0]


    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    annotated_frame = result.plot()


    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    detection_count = 0

    class_counts = {}


    if result.boxes is not None:

        detection_count = len(
            result.boxes
        )


        for cls in result.boxes.cls:

            class_id = int(
                cls.item()
            )


            # Get class name

            if hasattr(
                result,
                "names"
            ):

                class_name = result.names.get(
                    class_id,
                    str(class_id)
                )

            else:

                class_name = str(
                    class_id
                )


            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {

        "model": model_name,

        "detections": detection_count,

        "class_counts": class_counts,

        "inference_ms": round(
            inference_time * 1000,
            2
        ),

        "confidence": confidence,

        "iou": iou,

        "device": str(DEVICE),
    }


    return annotated_frame, stats


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    frame,
    extension=".jpg"
):

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )


    output_path = (
        RESULTS_DIR / filename
    )


    cv2.imwrite(
        str(output_path),
        frame
    )


    return filename


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def index(
    request: Request
):

    model_list = []


    for name, path in MODEL_PATHS.items():

        model_list.append(
            {
                "name": name,
                "available": path.is_file()
            }
        )


    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": model_list,
            "default_model": DEFAULT_MODEL
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    available_models = []


    for name, path in MODEL_PATHS.items():

        available_models.append(
            {
                "name": name,
                "available": path.is_file(),
                "path": str(path)
            }
        )


    return {

        "status": "ok",

        "device": str(DEVICE),

        "cuda_available":
            torch.cuda.is_available(),

        "models":
            available_models,

        "loaded_model":
            _loaded_model_name,
    }


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.post("/api/image")
async def detect_image(
    file: UploadFile = File(...),

    model: str = Form(DEFAULT_MODEL),

    confidence: float = Form(
        DEFAULT_CONFIDENCE
    ),

    iou: float = Form(
        DEFAULT_IOU
    )
):

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if model not in MODEL_PATHS:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}"
        )


    # --------------------------------------------------------
    # Validate confidence / IoU
    # --------------------------------------------------------

    confidence = get_float(
        confidence,
        DEFAULT_CONFIDENCE,
        0.0,
        1.0
    )


    iou = get_float(
        iou,
        DEFAULT_IOU,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )


    if len(contents) > MAX_CONTENT_LENGTH:

        raise HTTPException(
            status_code=413,
            detail="File is too large"
        )


    # --------------------------------------------------------
    # Convert bytes -> numpy array
    # --------------------------------------------------------

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    try:

        annotated_frame, stats = detect_frame(
            frame,
            model,
            confidence,
            iou
        )

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Encode image
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_frame
    )


    if not success:

        raise HTTPException(
            status_code=500,
            detail="Could not encode result image"
        )


    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    filename = save_image(
        annotated_frame
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "success": True,

        "filename": filename,

        "url": f"/results/{filename}",

        "image": (
            "data:image/jpeg;base64,"
            + image_base64
        ),

        "stats": stats,
    }


# ============================================================
# WEBCAM FRAME DETECTION
# ============================================================

@app.post("/api/frame")
async def detect_webcam_frame(
    image: str = Form(...),

    model: str = Form(DEFAULT_MODEL),

    confidence: float = Form(
        DEFAULT_CONFIDENCE
    ),

    iou: float = Form(
        DEFAULT_IOU
    )
):

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if model not in MODEL_PATHS:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}"
        )


    confidence = get_float(
        confidence,
        DEFAULT_CONFIDENCE,
        0.0,
        1.0
    )


    iou = get_float(
        iou,
        DEFAULT_IOU,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Remove data URL prefix
    # --------------------------------------------------------

    if "," in image:

        image = image.split(
            ",",
            1
        )[1]


    # --------------------------------------------------------
    # Decode Base64
    # --------------------------------------------------------

    try:

        image_bytes = base64.b64decode(
            image
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 image"
        )


    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode webcam frame"
        )


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    try:

        annotated_frame, stats = detect_frame(
            frame,
            model,
            confidence,
            iou
        )

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Encode result
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_frame
    )


    if not success:

        raise HTTPException(
            status_code=500,
            detail="Could not encode webcam result"
        )


    result_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


    return {

        "success": True,

        "image": (
            "data:image/jpeg;base64,"
            + result_base64
        ),

        "stats": stats,
    }


# ============================================================
# VIDEO DETECTION
# ============================================================

@app.post("/api/video")
async def detect_video(
    file: UploadFile = File(...),

    model: str = Form(DEFAULT_MODEL),

    confidence: float = Form(
        DEFAULT_CONFIDENCE
    ),

    iou: float = Form(
        DEFAULT_IOU
    )
):

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if model not in MODEL_PATHS:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}"
        )


    confidence = get_float(
        confidence,
        DEFAULT_CONFIDENCE,
        0.0,
        1.0
    )


    iou = get_float(
        iou,
        DEFAULT_IOU,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Read uploaded video
    # --------------------------------------------------------

    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Empty video file"
        )


    if len(contents) > MAX_CONTENT_LENGTH:

        raise HTTPException(
            status_code=413,
            detail="Video is too large"
        )


    # --------------------------------------------------------
    # Temporary input file
    # --------------------------------------------------------

    input_filename = (
        f"input_{uuid.uuid4().hex}.mp4"
    )


    input_path = (
        RESULTS_DIR / input_filename
    )


    with open(
        input_path,
        "wb"
    ) as f:

        f.write(contents)


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )


    if not cap.isOpened():

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="Could not open video"
        )


    # --------------------------------------------------------
    # Video properties
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 25.0


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    output_filename = (
        f"result_{uuid.uuid4().hex}.mp4"
    )


    output_path = (
        RESULTS_DIR / output_filename
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )


    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_count = 0

    total_detections = 0

    class_counts = {}

    start_time = time.perf_counter()


    try:

        while True:

            success, frame = cap.read()


            if not success:

                break


            # ----------------------------------------------
            # Detection
            # ----------------------------------------------

            annotated_frame, stats = detect_frame(
                frame,
                model,
                confidence,
                iou
            )


            # ----------------------------------------------
            # Write frame
            # ----------------------------------------------

            writer.write(
                annotated_frame
            )


            # ----------------------------------------------
            # Statistics
            # ----------------------------------------------

            frame_count += 1

            total_detections += stats[
                "detections"
            ]


            for class_name, count in stats[
                "class_counts"
            ].items():

                class_counts[class_name] = (
                    class_counts.get(
                        class_name,
                        0
                    ) + count
                )


    finally:

        cap.release()

        writer.release()


        # Remove temporary input

        input_path.unlink(
            missing_ok=True
        )


    # --------------------------------------------------------
    # Processing time
    # --------------------------------------------------------

    processing_time = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "success": True,

        "filename": output_filename,

        "url": (
            f"/results/{output_filename}"
        ),

        "stats": {

            "model": model,

            "frames": frame_count,

            "detections":
                total_detections,

            "class_counts":
                class_counts,

            "processing_time_seconds":
                round(
                    processing_time,
                    2
                ),

            "fps":
                round(
                    frame_count / processing_time,
                    2
                )
                if processing_time > 0
                else 0,

            "confidence":
                confidence,

            "iou":
                iou,

            "device":
                str(DEVICE),
        }
    }


# ============================================================
# SERVE RESULT FILES
# ============================================================

@app.get(
    "/results/{filename}"
)
async def get_result(
    filename: str
):

    file_path = (
        RESULTS_DIR / filename
    )


    # --------------------------------------------------------
    # Security check
    # --------------------------------------------------------

    try:

        file_path.resolve().relative_to(
            RESULTS_DIR.resolve()
        )

    except ValueError:

        raise HTTPException(
            status_code=403,
            detail="Invalid file path"
        )


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not file_path.is_file():

        raise HTTPException(
            status_code=404,
            detail="Result file not found"
        )


    return FileResponse(
        path=str(file_path)
    )


# ============================================================
# APPLICATION START MESSAGE
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main_fastapi:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )