"""
Application configuration.

All tunable parameters for the face recognition pipeline and attendance
business rules live here so the system can be re-tuned for a new office /
camera / lighting environment without touching the core logic.
"""

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loads variables from a .env file in the project root, if present, into
# os.environ — so `SECRET_KEY=...` etc. in .env are picked up below without
# needing to export them manually. Safe no-op if .env doesn't exist (e.g.
# in a deployment that sets real environment variables directly).
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DB_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ------------------------------------------------------------------ #
    # Storage paths
    # ------------------------------------------------------------------ #
    DATASET_DIR = os.path.join(BASE_DIR, "database", "dataset")        # raw enrolment images per employee
    MODEL_PATH = os.path.join(BASE_DIR, "database", "lbph_model.yml")  # trained recognizer
    LABELS_PATH = os.path.join(BASE_DIR, "database", "labels.json")    # label <-> employee_id map
    UNKNOWN_LOG_DIR = os.path.join(BASE_DIR, "static", "captures", "unknown")
    RECOGNIZED_LOG_DIR = os.path.join(BASE_DIR, "static", "captures", "recognized")

    # ------------------------------------------------------------------ #
    # Face detection (Haar Cascade — fast, dependency-free, CPU friendly)
    # ------------------------------------------------------------------ #
    # Bundled in the repo (database/cascades/) so the project doesn't depend
    # on cv2's optional `data` submodule being present in every OpenCV build;
    # falls back to the cv2-provided copy if available instead.
    HAAR_CASCADE = "haarcascade_frontalface_default.xml"
    BUNDLED_CASCADE_PATH = os.path.join(BASE_DIR, "database", "cascades", HAAR_CASCADE)
    DETECT_SCALE_FACTOR = 1.1
    DETECT_MIN_NEIGHBORS = 6
    DETECT_MIN_SIZE = (80, 80)          # ignore tiny / far-away faces (reduces false positives)
    MAX_FACES_PER_FRAME = 10            # safety cap for crowded frames

    # ------------------------------------------------------------------ #
    # Enrolment (registration)
    # ------------------------------------------------------------------ #
    SAMPLES_PER_EMPLOYEE = 30           # frames captured during registration
    FACE_IMG_SIZE = (200, 200)          # normalized size fed to the recognizer

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #
    # LBPH returns a *distance* (lower = more confident match).
    # Anything above this is treated as "Unknown". Tune per-deployment:
    # start around 70-85 for a webcam in a normal office.
    LBPH_CONFIDENCE_THRESHOLD = 75.0

    # A match is only accepted after N consecutive frames agree, which
    # filters out one-off misclassifications from a moving/partial face.
    RECOGNITION_CONSENSUS_FRAMES = 3

    # ------------------------------------------------------------------ #
    # Attendance business rules
    # ------------------------------------------------------------------ #
    # Once checked in, ignore repeat recognitions for this many minutes
    # (prevents "duplicate attendance" from someone lingering near the camera).
    DUPLICATE_COOLDOWN_MINUTES = 5

    # Minimum gap between check-in and an eligible check-out, in minutes.
    MIN_MINUTES_BEFORE_CHECKOUT = 60

    # Shift start used to flag "Late" (24h HH:MM).
    SHIFT_START_TIME = "09:30"
    LATE_GRACE_MINUTES = 15

    # ------------------------------------------------------------------ #
    # Lighting normalization
    # ------------------------------------------------------------------ #
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
