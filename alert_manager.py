from __future__ import annotations

import math
import threading
import wave
from pathlib import Path

from config import SOUNDS_DIR
from risk_classifier import HIGH, LOW, MEDIUM


class AlertManager:
    def __init__(self, sounds_dir: Path = SOUNDS_DIR) -> None:
        self.sounds_dir = sounds_dir
        self.high_sound_path = self.sounds_dir / "high_risk.wav"
        self.medium_sound_path = self.sounds_dir / "medium_risk.wav"
        self.current_level = LOW
        self._lock = threading.Lock()
        self._medium_thread: threading.Thread | None = None
        self._medium_stop = threading.Event()
        self._mixer_ready = False
        self._winsound_ready = False
        self._high_channel = None
        self._medium_sound = None
        self._high_sound = None
        self._high_stop = threading.Event()
        self._high_thread: threading.Thread | None = None

        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_default_sounds()
        self._init_mixer()

    def _ensure_default_sounds(self) -> None:
        if not self.high_sound_path.exists():
            self._write_tone(self.high_sound_path, frequency=880, duration=0.5)
        if not self.medium_sound_path.exists():
            self._write_tone(self.medium_sound_path, frequency=520, duration=0.18)

    @staticmethod
    def _write_tone(path: Path, frequency: int, duration: float) -> None:
        sample_rate = 44100
        amplitude = 16000
        frame_count = int(sample_rate * duration)

        with wave.open(str(path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for i in range(frame_count):
                value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                frames.extend(value.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(bytes(frames))

    def _init_mixer(self) -> None:
        try:
            import pygame

            pygame.mixer.init()
            self._high_sound = pygame.mixer.Sound(str(self.high_sound_path))
            self._medium_sound = pygame.mixer.Sound(str(self.medium_sound_path))
            self._mixer_ready = True
        except Exception:
            self._mixer_ready = False
            try:
                import winsound  # noqa: F401

                self._winsound_ready = True
            except Exception:
                self._winsound_ready = False

    def play_high_risk_alarm(self) -> None:
        self._stop_medium_loop()
        if self._mixer_ready and (
            self._high_channel is None or not self._high_channel.get_busy()
        ):
            self._high_channel = self._high_sound.play(loops=-1)
            return

        if self._winsound_ready and not (
            self._high_thread and self._high_thread.is_alive()
        ):
            self._high_stop.clear()
            self._high_thread = threading.Thread(target=self._high_loop, daemon=True)
            self._high_thread.start()

    def play_medium_risk_beep(self) -> None:
        self._stop_high_alarm()
        if self._medium_thread and self._medium_thread.is_alive():
            return

        self._medium_stop.clear()
        self._medium_thread = threading.Thread(target=self._medium_loop, daemon=True)
        self._medium_thread.start()

    def _medium_loop(self) -> None:
        while not self._medium_stop.is_set():
            if self._mixer_ready and self._medium_sound:
                self._medium_sound.play()
            elif self._winsound_ready:
                import winsound

                winsound.PlaySound(
                    str(self.medium_sound_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            self._medium_stop.wait(1.0)

    def _high_loop(self) -> None:
        import winsound

        while not self._high_stop.is_set():
            winsound.PlaySound(
                str(self.high_sound_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
            self._high_stop.wait(0.5)

    def stop_alarm(self) -> None:
        self._stop_high_alarm()
        self._stop_medium_loop()

    def _stop_high_alarm(self) -> None:
        if self._high_channel is not None:
            self._high_channel.stop()
            self._high_channel = None
        self._high_stop.set()
        self._high_thread = None
        if self._winsound_ready:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)

    def _stop_medium_loop(self) -> None:
        self._medium_stop.set()
        self._medium_thread = None
        if self._winsound_ready:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)

    def update(self, risk_level: str) -> None:
        with self._lock:
            if risk_level == self.current_level:
                return
            self.current_level = risk_level

            if risk_level == HIGH:
                self.play_high_risk_alarm()
            elif risk_level == MEDIUM:
                self.play_medium_risk_beep()
            else:
                self.stop_alarm()

    def close(self) -> None:
        self.stop_alarm()
        if self._mixer_ready:
            try:
                import pygame

                pygame.mixer.quit()
            except Exception:
                pass


_default_manager: AlertManager | None = None


def _manager() -> AlertManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


def play_high_risk_alarm() -> None:
    _manager().play_high_risk_alarm()


def play_medium_risk_beep() -> None:
    _manager().play_medium_risk_beep()


def stop_alarm() -> None:
    _manager().stop_alarm()
