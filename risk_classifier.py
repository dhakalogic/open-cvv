from dataclasses import dataclass


HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RiskResult:
    level: str
    color: tuple[int, int, int]
    ui_color: str
    message: str

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "color": self.color,
            "ui_color": self.ui_color,
            "message": self.message,
        }


def classify_risk(distance: float | None) -> dict:
    if distance is None or distance <= 0:
        return RiskResult(UNKNOWN, (180, 180, 180), "#9E9E9E", "No distance data").as_dict()

    if distance <= 2:
        return RiskResult(HIGH, (0, 0, 255), "#EF4444", "Human too close").as_dict()

    if distance <= 5:
        return RiskResult(MEDIUM, (0, 215, 255), "#FACC15", "Maintain distance").as_dict()

    return RiskResult(LOW, (0, 180, 0), "#22C55E", "Safe distance").as_dict()
