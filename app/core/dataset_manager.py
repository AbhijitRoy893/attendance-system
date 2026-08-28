"""
DatasetManager
==============
Owns the on-disk face dataset used to train the LBPH recognizer:

    database/dataset/<employee_id>/0001.png, 0002.png, ...

Kept separate from FaceEngine so storage concerns (folders, file naming,
re-loading the whole dataset for a retrain) don't leak into the CV logic.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import cv2
import numpy as np

from app.config import Config


class DatasetManager:
    def __init__(self, config: Config = Config):
        self.cfg = config
        os.makedirs(config.DATASET_DIR, exist_ok=True)

    def employee_dir(self, employee_id: int) -> str:
        path = os.path.join(self.cfg.DATASET_DIR, str(employee_id))
        os.makedirs(path, exist_ok=True)
        return path

    def save_sample(self, employee_id: int, normalized_face: np.ndarray) -> str:
        """Persist one normalized (already preprocessed) face crop."""
        folder = self.employee_dir(employee_id)
        existing = glob.glob(os.path.join(folder, "*.png"))
        next_idx = len(existing) + 1
        path = os.path.join(folder, f"{next_idx:04d}.png")
        cv2.imwrite(path, normalized_face)
        return path

    def count_samples(self, employee_id: int) -> int:
        return len(glob.glob(os.path.join(self.employee_dir(employee_id), "*.png")))

    def load_all(self) -> Dict[int, List[np.ndarray]]:
        """Load the entire dataset for a full retrain: {employee_id: [imgs]}"""
        dataset: Dict[int, List[np.ndarray]] = {}
        if not os.path.isdir(self.cfg.DATASET_DIR):
            return dataset

        for employee_folder in os.listdir(self.cfg.DATASET_DIR):
            folder_path = os.path.join(self.cfg.DATASET_DIR, employee_folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                employee_id = int(employee_folder)
            except ValueError:
                continue

            images = []
            for img_path in sorted(glob.glob(os.path.join(folder_path, "*.png"))):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
            if images:
                dataset[employee_id] = images

        return dataset

    def delete_employee(self, employee_id: int) -> None:
        folder = self.employee_dir(employee_id)
        for f in glob.glob(os.path.join(folder, "*.png")):
            os.remove(f)
        try:
            os.rmdir(folder)
        except OSError:
            pass
