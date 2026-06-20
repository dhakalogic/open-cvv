from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from config import MODEL_PATH


class HumanDetector:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        confidence_threshold: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        model_source = str(self.model_path if self.model_path.exists() else "yolov8n.pt")
        return YOLO(model_source)

    def detect_humans(self, frame) -> list[dict]:
        results = self.model(frame, verbose=False, classes=[0])
        detections: list[dict] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "width_pixels": max(1, x2 - x1),
                    }
                )

        return detections


def detect_humans(frame, detector: HumanDetector | None = None) -> list[dict]:
    active_detector = detector or HumanDetector()
    return active_detector.detect_humans(frame)


def draw_detection(frame, detection: dict, distance: float | None, risk: dict) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    color = risk["color"]
    confidence = detection["confidence"] * 100
    distance_text = "N/A" if distance is None else f"{distance:.2f} m"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{risk['level']} | {distance_text} | {confidence:.0f}%"
    label_y = max(24, y1 - 10)
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
