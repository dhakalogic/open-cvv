import json
from dataclasses import dataclass
from pathlib import Path

from config import (
    CALIBRATION_FILE,
    DEFAULT_FOCAL_LENGTH_PIXELS,
    DEFAULT_REFERENCE_DISTANCE_METERS,
    KNOWN_PERSON_WIDTH_METERS,
)


def calculate_focal_length(
    measured_distance: float,
    real_width: float,
    width_in_reference_image: float,
) -> float:
    if measured_distance <= 0:
        raise ValueError("Measured distance must be greater than zero.")
    if real_width <= 0:
        raise ValueError("Real object width must be greater than zero.")
    if width_in_reference_image <= 0:
        raise ValueError("Reference pixel width must be greater than zero.")

    return (width_in_reference_image * measured_distance) / real_width


def estimate_distance(
    pixel_width: float,
    focal_length: float = DEFAULT_FOCAL_LENGTH_PIXELS,
    real_width: float = KNOWN_PERSON_WIDTH_METERS,
) -> float | None:
    if pixel_width <= 0 or focal_length <= 0 or real_width <= 0:
        return None

    return (real_width * focal_length) / pixel_width


@dataclass
class CalibrationData:
    focal_length_pixels: float = DEFAULT_FOCAL_LENGTH_PIXELS
    known_person_width_meters: float = KNOWN_PERSON_WIDTH_METERS
    reference_distance_meters: float = DEFAULT_REFERENCE_DISTANCE_METERS


class DistanceEstimator:
    def __init__(self, calibration_file: Path = CALIBRATION_FILE) -> None:
        self.calibration_file = calibration_file
        self.data = self.load_calibration()

    @property
    def focal_length(self) -> float:
        return self.data.focal_length_pixels

    def load_calibration(self) -> CalibrationData:
        if not self.calibration_file.exists():
            return CalibrationData()

        try:
            payload = json.loads(self.calibration_file.read_text(encoding="utf-8"))
            return CalibrationData(
                focal_length_pixels=float(
                    payload.get("focal_length_pixels", DEFAULT_FOCAL_LENGTH_PIXELS)
                ),
                known_person_width_meters=float(
                    payload.get("known_person_width_meters", KNOWN_PERSON_WIDTH_METERS)
                ),
                reference_distance_meters=float(
                    payload.get(
                        "reference_distance_meters",
                        DEFAULT_REFERENCE_DISTANCE_METERS,
                    )
                ),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return CalibrationData()

    def save_calibration(self) -> None:
        payload = {
            "focal_length_pixels": self.data.focal_length_pixels,
            "known_person_width_meters": self.data.known_person_width_meters,
            "reference_distance_meters": self.data.reference_distance_meters,
        }
        self.calibration_file.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def calibrate(
        self,
        reference_pixel_width: float,
        reference_distance_meters: float = DEFAULT_REFERENCE_DISTANCE_METERS,
        real_width_meters: float = KNOWN_PERSON_WIDTH_METERS,
    ) -> float:
        focal_length = calculate_focal_length(
            reference_distance_meters,
            real_width_meters,
            reference_pixel_width,
        )
        self.data = CalibrationData(
            focal_length_pixels=focal_length,
            known_person_width_meters=real_width_meters,
            reference_distance_meters=reference_distance_meters,
        )
        self.save_calibration()
        return focal_length

    def estimate(self, pixel_width: float) -> float | None:
        return estimate_distance(
            pixel_width,
            self.data.focal_length_pixels,
            self.data.known_person_width_meters,
        )
