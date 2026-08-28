"""Small helpers for turning browser webcam frames into OpenCV images."""

import base64
import os
import re
from datetime import datetime

import cv2
import numpy as np


def decode_base64_image(data_url: str) -> np.ndarray:
    """Convert a `data:image/jpeg;base64,...` string from the browser
    <canvas> capture into an OpenCV BGR numpy array."""
    match = re.match(r"data:image/\w+;base64,(.*)", data_url)
    b64_data = match.group(1) if match else data_url
    img_bytes = base64.b64decode(b64_data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data")
    return frame


def save_snapshot(frame_bgr: np.ndarray, directory: str, prefix: str = "capture") -> str:
    os.makedirs(directory, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def draw_annotations(frame_bgr: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes + labels for recognized/unknown faces (used by
    the optional debug/preview endpoint)."""
    annotated = frame_bgr.copy()
    for r in results:
        x, y, w, h = r.box
        color = (0, 200, 0) if r.is_known else (0, 0, 255)
        label = f"ID:{r.employee_id} ({r.confidence:.0f})" if r.is_known else "Unknown"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated
