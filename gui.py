from __future__ import annotations

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, TOP, Button, Frame, Label, Tk

import cv2
from PIL import Image, ImageTk

from alert_manager import AlertManager
from config import (
    APP_TITLE,
    CAMERA_INDEX,
    DETECTION_LOG,
    RECORDINGS_DIR,
    SCREENSHOTS_DIR,
    TARGET_FPS,
    ensure_directories,
)
from detector import HumanDetector, draw_detection
from distance_estimator import DistanceEstimator
from risk_classifier import HIGH, LOW, MEDIUM, UNKNOWN, classify_risk


class DetectionLogger:
    def __init__(self, path: Path = DETECTION_LOG) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "distance_m", "risk_level", "confidence"])

    def write(self, distance: float | None, risk_level: str, confidence: float) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    "" if distance is None else f"{distance:.3f}",
                    risk_level,
                    f"{confidence:.4f}",
                ]
            )


class DashboardApp:
    def __init__(self) -> None:
        ensure_directories()

        self.root = Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1280x760")
        self.root.configure(bg="#121212")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.detector = HumanDetector()
        self.estimator = DistanceEstimator()
        self.alerts = AlertManager()
        self.logger = DetectionLogger()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.running = True
        self.recording = False
        self.video_writer = None
        self.last_log_time = 0.0
        self.last_frame = None
        self.frame_times = deque(maxlen=30)

        self.total_humans_detected = 0
        self.distances: list[float] = []
        self.risk_counts = {HIGH: 0, MEDIUM: 0, LOW: 0}

        self._build_ui()
        self._bind_keys()

    def _build_ui(self) -> None:
        header = Frame(self.root, bg="#121212", padx=18, pady=14)
        header.pack(side=TOP, fill="x")

        title = Label(
            header,
            text="Human Distance Estimation",
            fg="white",
            bg="#121212",
            font=("Segoe UI", 22, "bold"),
        )
        title.pack(side=LEFT)

        self.clock_label = Label(
            header,
            text="",
            fg="#CFCFCF",
            bg="#121212",
            font=("Segoe UI", 12),
        )
        self.clock_label.pack(side=RIGHT)

        body = Frame(self.root, bg="#121212", padx=18, pady=6)
        body.pack(fill=BOTH, expand=True)

        feed_panel = Frame(body, bg="#1E1E1E", padx=10, pady=10)
        feed_panel.pack(side=LEFT, fill=BOTH, expand=True)

        self.video_label = Label(feed_panel, bg="#0B0B0B")
        self.video_label.pack(fill=BOTH, expand=True)

        controls = Frame(feed_panel, bg="#1E1E1E", pady=10)
        controls.pack(fill="x")

        Button(controls, text="Calibrate (C)", command=self.calibrate_current_frame).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Screenshot (S)", command=self.capture_screenshot).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Record (R)", command=self.toggle_recording).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Quit (Q)", command=self.close).pack(side=RIGHT, padx=4)

        side_panel = Frame(body, bg="#121212", padx=14)
        side_panel.pack(side=RIGHT, fill="y")

        self.cards: dict[str, Label] = {}
        for key, label in [
            ("status", "Status"),
            ("human_count", "Human Count"),
            ("distance", "Nearest Distance"),
            ("risk", "Risk Level"),
            ("fps", "FPS"),
            ("confidence", "Confidence"),
            ("recording", "Recording"),
            ("total", "Total Humans Detected"),
            ("average", "Average Distance"),
            ("risk_counts", "Risk Counts"),
        ]:
            self._add_card(side_panel, key, label)

    def _add_card(self, parent: Frame, key: str, label: str) -> None:
        card = Frame(parent, bg="#1E1E1E", padx=14, pady=10)
        card.pack(fill="x", pady=6)
        Label(
            card,
            text=label,
            fg="#A3A3A3",
            bg="#1E1E1E",
            font=("Segoe UI", 10),
        ).pack(anchor="w")
        value = Label(
            card,
            text="-",
            fg="white",
            bg="#1E1E1E",
            font=("Segoe UI", 16, "bold"),
            wraplength=300,
            justify=LEFT,
        )
        value.pack(anchor="w")
        self.cards[key] = value

    def _bind_keys(self) -> None:
        self.root.bind("<c>", lambda _event: self.calibrate_current_frame())
        self.root.bind("<C>", lambda _event: self.calibrate_current_frame())
        self.root.bind("<s>", lambda _event: self.capture_screenshot())
        self.root.bind("<S>", lambda _event: self.capture_screenshot())
        self.root.bind("<r>", lambda _event: self.toggle_recording())
        self.root.bind("<R>", lambda _event: self.toggle_recording())
        self.root.bind("<q>", lambda _event: self.close())
        self.root.bind("<Q>", lambda _event: self.close())

    def run(self) -> None:
        self._update_frame()
        self.root.mainloop()

    def _update_frame(self) -> None:
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.cards["status"].config(text="Camera unavailable", fg="#EF4444")
            self.root.after(500, self._update_frame)
            return

        self.last_frame = frame.copy()
        started = time.perf_counter()
        detections = self.detector.detect_humans(frame)
        nearest_distance = None
        nearest_confidence = 0.0
        highest_risk = classify_risk(None)

        self.total_humans_detected += len(detections)

        for detection in detections:
            distance = self.estimator.estimate(detection["width_pixels"])
            risk = classify_risk(distance)
            draw_detection(frame, detection, distance, risk)

            if distance is not None:
                self.distances.append(distance)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_confidence = detection["confidence"]
                    highest_risk = risk

        if detections and highest_risk["level"] in self.risk_counts:
            self.risk_counts[highest_risk["level"]] += 1

        self._draw_overlay(frame, len(detections), nearest_distance, highest_risk)
        self._write_recording_frame(frame)
        self._log_detection(nearest_distance, highest_risk["level"], nearest_confidence)
        self.alerts.update(highest_risk["level"])

        elapsed = time.perf_counter() - started
        self.frame_times.append(elapsed)
        fps = 0.0 if not self.frame_times else 1 / (sum(self.frame_times) / len(self.frame_times))
        self._update_cards(
            human_count=len(detections),
            distance=nearest_distance,
            risk=highest_risk,
            fps=fps,
            confidence=nearest_confidence,
        )
        self._show_frame(frame)

        delay = max(1, int(1000 / TARGET_FPS))
        self.root.after(delay, self._update_frame)

    def _draw_overlay(
        self,
        frame,
        human_count: int,
        nearest_distance: float | None,
        risk: dict,
    ) -> None:
        distance_text = "N/A" if nearest_distance is None else f"{nearest_distance:.2f} m"
        lines = [
            f"Human Count: {human_count}",
            f"Distance: {distance_text}",
            f"Risk: {risk['level']}",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 32

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        panel_width = max(640, self.video_label.winfo_width())
        panel_height = max(420, self.video_label.winfo_height())
        image.thumbnail((panel_width, panel_height))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def _update_cards(
        self,
        human_count: int,
        distance: float | None,
        risk: dict,
        fps: float,
        confidence: float,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_distance = sum(self.distances) / len(self.distances) if self.distances else 0.0

        self.clock_label.config(text=now)
        self.cards["status"].config(text=risk["message"], fg=risk["ui_color"])
        self.cards["human_count"].config(text=str(human_count), fg="white")
        self.cards["distance"].config(
            text="-" if distance is None else f"{distance:.2f} m",
            fg="white",
        )
        self.cards["risk"].config(text=risk["level"], fg=risk["ui_color"])
        self.cards["fps"].config(text=f"{fps:.1f}", fg="white")
        self.cards["confidence"].config(text=f"{confidence * 100:.0f}%", fg="white")
        self.cards["recording"].config(
            text="ON" if self.recording else "OFF",
            fg="#EF4444" if self.recording else "#A3A3A3",
        )
        self.cards["total"].config(text=str(self.total_humans_detected), fg="white")
        self.cards["average"].config(
            text="-" if not self.distances else f"{avg_distance:.2f} m",
            fg="white",
        )
        self.cards["risk_counts"].config(
            text=(
                f"High: {self.risk_counts[HIGH]}\n"
                f"Medium: {self.risk_counts[MEDIUM]}\n"
                f"Low: {self.risk_counts[LOW]}"
            ),
            fg="white",
        )

    def _log_detection(
        self,
        distance: float | None,
        risk_level: str,
        confidence: float,
    ) -> None:
        if risk_level == UNKNOWN:
            return
        now = time.time()
        if now - self.last_log_time < 1:
            return
        self.logger.write(distance, risk_level, confidence)
        self.last_log_time = now

    def capture_screenshot(self) -> None:
        if self.last_frame is None:
            return
        filename = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        cv2.imwrite(str(SCREENSHOTS_DIR / filename), self.last_frame)

    def calibrate_current_frame(self) -> None:
        if self.last_frame is None:
            return
        detections = self.detector.detect_humans(self.last_frame)
        if not detections:
            self.cards["status"].config(text="No human found for calibration", fg="#FACC15")
            return

        largest = max(detections, key=lambda item: item["width_pixels"])
        self.estimator.calibrate(largest["width_pixels"])
        self.cards["status"].config(text="Calibration saved", fg="#22C55E")

    def toggle_recording(self) -> None:
        if self.recording:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.avi")
        output_path = RECORDINGS_DIR / filename
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))
        self.recording = True

    def _write_recording_frame(self, frame) -> None:
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def close(self) -> None:
        self.running = False
        self.alerts.close()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
