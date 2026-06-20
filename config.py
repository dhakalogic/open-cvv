from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
SOUNDS_DIR = BASE_DIR / "sounds"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
RECORDINGS_DIR = BASE_DIR / "recordings"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CALIBRATION_FILE = BASE_DIR / "calibration.json"
DETECTION_LOG = LOGS_DIR / "detections.csv"

KNOWN_PERSON_WIDTH_METERS = 0.45
DEFAULT_REFERENCE_DISTANCE_METERS = 2.0
DEFAULT_FOCAL_LENGTH_PIXELS = 700.0

CAMERA_INDEX = 0
TARGET_FPS = 30

APP_TITLE = "Real-Time Human Distance Estimation and Risk Alert System"


def ensure_directories() -> None:
    for directory in (
        MODELS_DIR,
        SOUNDS_DIR,
        SCREENSHOTS_DIR,
        RECORDINGS_DIR,
        LOGS_DIR,
        ASSETS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
