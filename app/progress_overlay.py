"""Non-activating corner window for live typing progress."""

from __future__ import annotations

import ctypes
import sys

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def reduced_motion_enabled() -> bool:
    """Honor Windows client-area animation preference when available."""
    if sys.platform != "win32":
        return False
    enabled = ctypes.c_int(1)
    try:
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0)
    except (AttributeError, OSError):
        return False
    return not bool(enabled.value)


class ProgressOverlay(QWidget):
    stop_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    open_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("pytyper progress")
        self.setFixedSize(316, 190)
        self._reduce_motion = reduced_motion_enabled()
        self._entrance = QPropertyAnimation(self, b"pos", self)
        self._entrance.setDuration(260)
        self._entrance.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 13, 16, 12)
        root.setSpacing(9)
        header = QHBoxLayout()
        title = QLabel("pytyper")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self.eta_label = QLabel("--:-- left")
        header.addWidget(self.eta_label)
        root.addLayout(header)

        self.status_label = QLabel("Preparing typing…")
        self.status_label.setMaximumHeight(18)
        root.addWidget(self.status_label)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(0)
        self.wpm_label = self._metric(metrics, 0, "0", "WPM")
        self.accuracy_label = self._metric(metrics, 1, "100%", "ACCURACY")
        self.percent_label = self._metric(metrics, 2, "0%", "DONE")
        root.addLayout(metrics)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)
        self._progress_animation = QPropertyAnimation(self.progress_bar, b"value", self)
        self._progress_animation.setDuration(180)
        self._progress_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        actions = QHBoxLayout()
        actions.addStretch()
        self.open_button = QPushButton("Open app")
        self.open_button.clicked.connect(self.open_requested.emit)
        self.open_button.hide()
        actions.addWidget(self.open_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        actions.addWidget(self.pause_button)
        self.stop_button = QPushButton("Stop typing")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        actions.addWidget(self.stop_button)
        root.addLayout(actions)

    @staticmethod
    def _metric(layout: QGridLayout, column: int, value: str, caption: str) -> QLabel:
        number = QLabel(value)
        number.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        label = QLabel(caption)
        layout.addWidget(number, 0, column)
        layout.addWidget(label, 1, column)
        return number

    def show_running(self) -> None:
        self.status_label.setText("Preparing typing…")
        self.eta_label.setText("--:-- left")
        self.wpm_label.setText("0")
        self.accuracy_label.setText("100%")
        self.percent_label.setText("0%")
        self.progress_bar.setValue(0)
        self.open_button.hide()
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.show()
        self.stop_button.show()
        screen = QApplication.screenAt(self.cursor().pos()) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        corner = QPoint(area.right() - self.width() - 18, area.bottom() - self.height() - 18)
        self.move(corner if self._reduce_motion else corner + QPoint(28, 0))
        self.show()
        if not self._reduce_motion:
            self._entrance.stop()
            self._entrance.setStartValue(self.pos())
            self._entrance.setEndValue(corner)
            self._entrance.start()

    def set_status(self, message: str) -> None:
        if message.startswith("Loading"):
            self.status_label.setText("Loading model…")
        elif message.startswith("Model ready"):
            self.status_label.setText("Choose target window now")
        elif message.startswith("AI typing"):
            self.status_label.setText("Typing in target window")
            self.pause_button.setEnabled(True)
        elif message.startswith("Stopping"):
            self.status_label.setText(message)
            self.pause_button.setEnabled(False)
        else:
            self.status_label.setText(message)

    def _toggle_pause(self) -> None:
        if self.pause_button.text() == "Pause":
            self.pause_requested.emit()
            self.pause_button.setText("Resume")
            self.status_label.setText("Paused")
        else:
            self.resume_requested.emit()
            self.pause_button.setText("Pause")
            self.status_label.setText("Typing in target window")

    def update_progress(self, percent: int, wpm: float, accuracy: float, seconds_left: float) -> None:
        self.wpm_label.setText(str(round(wpm)))
        self.accuracy_label.setText(f"{accuracy:.0f}%")
        self.percent_label.setText(f"{percent}%")
        minutes, seconds = divmod(round(max(0, seconds_left)), 60)
        self.eta_label.setText(f"{minutes}:{seconds:02d} left")
        if self._reduce_motion or percent == self.progress_bar.value():
            self.progress_bar.setValue(percent)
            return
        self._progress_animation.stop()
        self._progress_animation.setStartValue(self.progress_bar.value())
        self._progress_animation.setEndValue(percent)
        self._progress_animation.start()

    def show_finished(self, message: str) -> None:
        self.status_label.setText(message)
        self.eta_label.setText("Done" if message == "Typing completed." else "Stopped")
        self.stop_button.hide()
        self.pause_button.hide()
        self.open_button.show()
