"""Background worker that emits generated keyboard events."""

from __future__ import annotations

import threading
import time

import keyboard
from PyQt6.QtCore import QObject, pyqtSignal

from app.config import TypingConfig
from app.progress import ProgressTracker

_RUNTIME_CACHE: dict[str, object] = {}
_RUNTIME_LOCK = threading.Lock()
_PRELOADING: set[str] = set()


def _runtime_for(model_path: str):
    """Import PyTorch on first use and retain the loaded model for later runs."""
    with _RUNTIME_LOCK:
        runtime = _RUNTIME_CACHE.get(model_path)
        if runtime is None:
            from model.runtime import NeuralTypingRuntime

            runtime = NeuralTypingRuntime(model_path)
            _RUNTIME_CACHE.clear()
            _RUNTIME_CACHE[model_path] = runtime
        return runtime


def preload_runtime(model_path: str) -> None:
    """Warm the model off the UI thread so first typing run can reuse it."""
    with _RUNTIME_LOCK:
        if model_path in _RUNTIME_CACHE or model_path in _PRELOADING:
            return
        _PRELOADING.add(model_path)

    def load() -> None:
        try:
            _runtime_for(model_path)
        except Exception:
            pass  # The typing run reports loading errors to the user.
        finally:
            with _RUNTIME_LOCK:
                _PRELOADING.discard(model_path)

    threading.Thread(target=load, name="pytyper model preload", daemon=True).start()


class TypingWorker(QObject):
    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(int, float, float, float)
    finished = pyqtSignal(bool, str)

    def __init__(self, config: TypingConfig) -> None:
        super().__init__()
        self._config = config
        self._stop_event = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False
        self._hotkey_id: int | None = None

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._pause_condition:
            self._pause_condition.notify_all()

    def request_pause(self) -> None:
        with self._pause_condition:
            self._paused = True
            self._pause_condition.notify_all()

    def request_resume(self) -> None:
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def run(self) -> None:
        canceled = False
        message = "Typing completed."
        try:
            self._hotkey_id = keyboard.add_hotkey("esc", self.request_stop)
        except Exception:
            self._hotkey_id = None

        try:
            self.status_changed.emit("Loading trained human typing model...")
            runtime = _runtime_for(self._config.model_path)
            events = runtime.generate(self._config.text, self._config.controls)
            self.status_changed.emit("Model ready. Switch to the target window.")
            if self._stop_event.wait(self._config.start_delay_ms / 1000.0):
                canceled, message = True, "Typing canceled before start."
                return

            self.status_changed.emit("AI typing started. Press ESC to stop.")
            tracker = ProgressTracker(self._config.text, sum(event.delay_before for event in events))
            self.progress_changed.emit(0, 0.0, 100.0, tracker.planned_seconds)
            canceled = not self._emit_events(events, tracker)
            if canceled:
                message = "Typing canceled."
        except Exception as error:
            canceled, message = True, f"Typing model error: {error}"
        finally:
            if self._hotkey_id is not None:
                try:
                    keyboard.remove_hotkey(self._hotkey_id)
                except Exception:
                    pass
            self.finished.emit(canceled, message)

    @staticmethod
    def _emit_key(key: str) -> None:
        special_keys = {"BACKSPACE": "backspace", "\n": "enter", "\t": "tab", " ": "space"}
        mapped = special_keys.get(key)
        if mapped:
            keyboard.press_and_release(mapped)
        else:
            keyboard.write(key, delay=0)

    def _emit_events(self, events, tracker: ProgressTracker | None = None) -> bool:
        """Schedule press/release events; overlapping holds produce rollover."""

        pending: list[tuple[float, str]] = []
        deadline = time.monotonic()
        try:
            for event in events:
                deadline += event.delay_before
                while pending:
                    due, key = min(pending)
                    if due > deadline:
                        break
                    resumed_due = self._wait_until(due, pending, tracker)
                    if resumed_due is None:
                        return False
                    deadline += resumed_due - due
                    if (due, key) in pending:
                        keyboard.release(key)
                        pending.remove((due, key))
                while True:
                    resumed_deadline = self._wait_until(deadline, pending, tracker)
                    if resumed_deadline is None:
                        return False
                    deadline = resumed_deadline
                    with self._pause_condition:
                        if self._paused:
                            continue
                        physical = self._config.controls.physical_keys and event.hold_seconds > 0
                        mapped = self._physical_key(event.key) if physical else None
                        if mapped is None:
                            # Text entry may use modifiers; release held keys first.
                            for _, key in pending:
                                keyboard.release(key)
                            pending.clear()
                            self._emit_key(event.key)
                        else:
                            for due, key in tuple(pending):
                                if key == mapped:
                                    keyboard.release(key)
                                    pending.remove((due, key))
                            keyboard.press(mapped)
                            pending.append((time.monotonic() + event.hold_seconds, mapped))
                        self._report_progress(tracker, event)
                        break
            for due, key in sorted(pending):
                if self._wait_until(due, pending, tracker) is None:
                    return False
                if (due, key) in pending:
                    keyboard.release(key)
                    pending.remove((due, key))
            return True
        finally:
            for _, key in pending:
                try:
                    keyboard.release(key)
                except Exception:
                    pass

    def _wait_until(
        self, deadline: float, pending: list[tuple[float, str]], tracker: ProgressTracker | None
    ) -> float | None:
        with self._pause_condition:
            while not self._stop_event.is_set():
                if self._paused:
                    for _, key in pending:
                        keyboard.release(key)
                    pending.clear()
                    paused_at = time.monotonic()
                    while self._paused and not self._stop_event.is_set():
                        self._pause_condition.wait()
                    paused_seconds = time.monotonic() - paused_at
                    deadline += paused_seconds
                    if tracker is not None:
                        tracker.add_pause(paused_seconds)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return deadline
                self._pause_condition.wait(timeout=remaining)
        return None

    def _report_progress(self, tracker: ProgressTracker | None, event) -> None:
        if tracker is None:
            return
        progress = tracker.advance(event.key, event.delay_before)
        self.progress_changed.emit(progress.percent, progress.wpm, progress.accuracy, progress.seconds_left)

    @staticmethod
    def _physical_key(key: str) -> str | None:
        if key == "BACKSPACE":
            return "backspace"
        if key == " ":
            return "space"
        if len(key) == 1 and ("a" <= key <= "z" or "0" <= key <= "9"):
            return key
        return None
