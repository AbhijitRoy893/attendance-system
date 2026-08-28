"""
Unit tests for the computer-vision layer (app/core/face_engine.py).

Uses a standard public-domain test image so the tests are self-contained
and don't require a live webcam.

Run with:  pytest tests/ -v
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.core.face_engine import FaceEngine

SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture(scope="module")
def engine():
    return FaceEngine(Config)


@pytest.fixture(scope="module")
def sample_frame():
    img = cv2.imread(SAMPLE_IMAGE)
    assert img is not None, f"Test fixture missing: {SAMPLE_IMAGE}"
    return img


def test_detects_face_in_sample_image(engine, sample_frame):
    faces = engine.detect_faces(sample_frame)
    assert len(faces) == 1
    x, y, w, h = faces[0].box
    assert w > 0 and h > 0


def test_normalized_face_has_expected_size(engine, sample_frame):
    faces = engine.detect_faces(sample_frame)
    assert faces[0].aligned.shape == Config.FACE_IMG_SIZE


def test_no_faces_in_blank_frame(engine):
    blank = np.zeros((480, 640, 3), dtype="uint8")
    faces = engine.detect_faces(blank)
    assert len(faces) == 0


def test_recognize_returns_unknown_when_untrained(sample_frame):
    fresh_engine = FaceEngine(Config)
    fresh_engine._model_loaded = False
    fresh_engine.labels = {}
    results = fresh_engine.recognize(sample_frame)
    assert all(not r.is_known for r in results)


def test_train_and_recognize_round_trip(sample_frame):
    engine = FaceEngine(Config)
    faces = engine.detect_faces(sample_frame)
    assert len(faces) == 1

    # Train on several near-duplicate samples of the same identity (id=1)
    samples = {1: [faces[0].aligned for _ in range(10)]}
    engine.train(samples)

    results = engine.recognize(sample_frame)
    assert len(results) == 1
    assert results[0].is_known is True
    assert results[0].employee_id == 1
