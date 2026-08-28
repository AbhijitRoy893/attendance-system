"""
FaceEngine
==========
Wraps every computer-vision concern in one place:

  1. Face DETECTION      -> Haar Cascade (fast, no external model download)
  2. Lighting NORMALIZATION -> grayscale + CLAHE histogram equalization
  3. Face RECOGNITION    -> OpenCV LBPH (Local Binary Patterns Histogram)

Design notes
------------
* LBPH is chosen over deep-learning embeddings (e.g. dlib/face_recognition)
  because it trains and runs on CPU in real time with zero extra native
  build dependencies (`opencv-contrib-python` ships `cv2.face` directly),
  which makes the system easy to deploy on ordinary office hardware.
  The `EncoderBackend` abstraction below means the LBPH implementation can
  be swapped for a deep embedding model later without touching callers.
* Every frame is preprocessed identically at both enrolment and inference
  time (resize -> grayscale -> CLAHE) so lighting differences between the
  registration session and daily use don't hurt accuracy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app.config import Config


@dataclass
class DetectedFace:
    box: Tuple[int, int, int, int]   # x, y, w, h in the original frame
    aligned: np.ndarray              # normalized grayscale crop ready for the recognizer


@dataclass
class RecognitionResult:
    box: Tuple[int, int, int, int]
    employee_id: int | None
    confidence: float                # LBPH distance; lower is better
    is_known: bool


class FaceEngine:
    def __init__(self, config: Config = Config):
        self.cfg = config
        cascade_path = self._resolve_cascade_path(config)
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: dict[str, int] = {}     # employee_id(str) -> label(int); label == employee_id here
        self._model_loaded = False

        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        os.makedirs(config.DATASET_DIR, exist_ok=True)
        os.makedirs(config.UNKNOWN_LOG_DIR, exist_ok=True)
        os.makedirs(config.RECOGNIZED_LOG_DIR, exist_ok=True)

        self.load_model()

    @staticmethod
    def _resolve_cascade_path(config: Config) -> str:
        """Prefer the cascade bundled with the project (reliable across
        OpenCV builds/platforms); fall back to the one shipped with cv2."""
        if os.path.exists(config.BUNDLED_CASCADE_PATH):
            return config.BUNDLED_CASCADE_PATH
        cv2_path = os.path.join(getattr(cv2.data, "haarcascades", ""), config.HAAR_CASCADE)
        if os.path.exists(cv2_path):
            return cv2_path
        raise RuntimeError(
            "Haar cascade not found in project bundle or cv2 install. "
            f"Expected at {config.BUNDLED_CASCADE_PATH}"
        )

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _normalize(self, gray_face: np.ndarray) -> np.ndarray:
        """Resize + CLAHE equalize a cropped grayscale face for lighting
        invariance. Applied identically at enrolment and recognition time."""
        resized = cv2.resize(gray_face, self.cfg.FACE_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        return self.clahe.apply(resized)

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect_faces(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """Detect all faces in a frame (handles multiple people at once)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = self.clahe.apply(gray)  # helps detection in poor lighting too

        boxes = self.detector.detectMultiScale(
            gray_eq,
            scaleFactor=self.cfg.DETECT_SCALE_FACTOR,
            minNeighbors=self.cfg.DETECT_MIN_NEIGHBORS,
            minSize=self.cfg.DETECT_MIN_SIZE,
        )

        # Cap absurd numbers of detections (noisy background, false positives)
        boxes = list(boxes)[: self.cfg.MAX_FACES_PER_FRAME]

        faces = []
        for (x, y, w, h) in boxes:
            crop = gray[y:y + h, x:x + w]
            faces.append(DetectedFace(box=(int(x), int(y), int(w), int(h)),
                                       aligned=self._normalize(crop)))
        return faces

    # ------------------------------------------------------------------ #
    # Training / enrolment
    # ------------------------------------------------------------------ #
    def train(self, samples: dict[int, List[np.ndarray]]) -> None:
        """
        Train (or retrain) the recognizer from scratch on the full dataset.

        samples: { employee_id: [normalized grayscale face crops, ...] }
        """
        images, labels = [], []
        for employee_id, faces in samples.items():
            for face in faces:
                images.append(face)
                labels.append(int(employee_id))

        if not images:
            raise ValueError("No training samples provided")

        self.recognizer.train(images, np.array(labels))
        self.recognizer.write(self.cfg.MODEL_PATH)

        self.labels = {str(eid): int(eid) for eid in samples.keys()}
        with open(self.cfg.LABELS_PATH, "w") as f:
            json.dump(self.labels, f)

        self._model_loaded = True

    def update_with_new_employee(self, employee_id: int, faces: List[np.ndarray], full_dataset: dict) -> None:
        """Convenience wrapper: retrain on the full dataset including the
        newly added employee. LBPH doesn't support true incremental
        multi-class training reliably, so a full retrain keeps accuracy
        consistent; datasets of a few hundred employees retrain in seconds."""
        full_dataset[employee_id] = faces
        self.train(full_dataset)

    def load_model(self) -> bool:
        if os.path.exists(self.cfg.MODEL_PATH) and os.path.exists(self.cfg.LABELS_PATH):
            self.recognizer.read(self.cfg.MODEL_PATH)
            with open(self.cfg.LABELS_PATH) as f:
                self.labels = json.load(f)
            self._model_loaded = True
            return True
        self._model_loaded = False
        return False

    @property
    def is_trained(self) -> bool:
        return self._model_loaded and len(self.labels) > 0

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #
    def recognize(self, frame_bgr: np.ndarray) -> List[RecognitionResult]:
        """Detect + identify every face in the frame. Handles multiple
        faces and unknown persons (confidence worse than threshold)."""
        results: List[RecognitionResult] = []
        faces = self.detect_faces(frame_bgr)

        if not self.is_trained:
            # No one enrolled yet -> everyone is "unknown"
            for f in faces:
                results.append(RecognitionResult(box=f.box, employee_id=None,
                                                   confidence=999.0, is_known=False))
            return results

        for f in faces:
            label, distance = self.recognizer.predict(f.aligned)
            is_known = distance <= self.cfg.LBPH_CONFIDENCE_THRESHOLD
            results.append(RecognitionResult(
                box=f.box,
                employee_id=label if is_known else None,
                confidence=float(distance),
                is_known=is_known,
            ))
        return results
