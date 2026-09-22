"""Compact PyQt6 controls for the trained typing model."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QSettings, QSignalBlocker, Qt, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import TypingConfig
from app.progress_overlay import ProgressOverlay, reduced_motion_enabled
from app.typing_worker import TypingWorker, preload_runtime
from model.controls import TypingControls


class PytyperWindow(QMainWindow):
    MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
    MODEL_PATH = MODEL_DIR / "pytyper.pt"
    MAX_FILE_BYTES = 2 * 1024 * 1024

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pytyper")
        self.resize(760, 610)
        self.setMinimumSize(660, 540)
        self._thread: QThread | None = None
        self._worker: TypingWorker | None = None
        self._settings = QSettings()
        self._source_name = "Draft"
        self.progress_overlay = ProgressOverlay()
        self.progress_overlay.stop_requested.connect(self._stop_typing)
        self.progress_overlay.pause_requested.connect(self._pause_typing)
        self.progress_overlay.resume_requested.connect(self._resume_typing)
        self.progress_overlay.open_requested.connect(self._show_main)
        self._build_ui()
        self._load_settings()
        if self.MODEL_PATH.exists():
            preload_runtime(str(self.MODEL_PATH))

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 15, 18, 14)
        layout.setSpacing(9)

        header = QHBoxLayout()
        header.setSpacing(10)
        title = QLabel("pytyper")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self.model_label = QLabel()
        header.addWidget(self.model_label)
        layout.addLayout(header)

        subtitle = QLabel("Realistic human typing on a small nn model")
        layout.addWidget(subtitle)

        source = QHBoxLayout()
        source.setSpacing(9)
        text_heading = QLabel("Text to type")
        source.addWidget(text_heading)
        self.source_label = QLabel("Draft · 0 characters")
        source.addWidget(self.source_label)
        source.addStretch()
        self.load_button = QPushButton("Load file…")
        self.load_button.setToolTip("Load a UTF-8 text file into the editor")
        self.load_button.clicked.connect(self._choose_file)
        source.addWidget(self.load_button)
        layout.addLayout(source)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Paste text here or load a UTF-8 file…")
        self.text_edit.setMinimumHeight(155)
        self.text_edit.setFont(QFont("Consolas", 12))
        self.text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.text_edit, 1)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)
        options_heading = QLabel("Typing style")
        layout.addWidget(options_heading)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.wpm_spin = QSpinBox()
        self.wpm_spin.setRange(20, 200)
        self.wpm_spin.setValue(75)
        self.wpm_spin.setSuffix(" WPM")
        self.wpm_spin.setToolTip("Approximate speed; five characters count as one word")
        self._field(grid, 0, 0, "Target speed", self.wpm_spin)

        self.error_spin = QDoubleSpinBox()
        self.error_spin.setRange(0.0, 25.0)
        self.error_spin.setDecimals(1)
        self.error_spin.setSingleStep(0.5)
        self.error_spin.setValue(3.5)
        self.error_spin.setSuffix(" %")
        self.error_spin.setToolTip("Approximate share of typed characters that are errors")
        self._field(grid, 0, 1, "Error rate", self.error_spin)

        self.rhythm_spin = QSpinBox()
        self.rhythm_spin.setRange(50, 150)
        self.rhythm_spin.setValue(100)
        self.rhythm_spin.setSuffix(" %")
        self.rhythm_spin.setToolTip("Lower is steadier; higher emphasizes bursts")
        self._field(grid, 1, 0, "Rhythm variation", self.rhythm_spin)

        self.pause_spin = QSpinBox()
        self.pause_spin.setRange(25, 200)
        self.pause_spin.setValue(100)
        self.pause_spin.setSuffix(" %")
        self.pause_spin.setToolTip("Changes how often the model pauses")
        self._field(grid, 1, 1, "Pause tendency", self.pause_spin)

        self.start_delay_spin = QSpinBox()
        self.start_delay_spin.setRange(0, 10000)
        self.start_delay_spin.setValue(1800)
        self.start_delay_spin.setSuffix(" ms")
        self.start_delay_spin.setToolTip("Time to focus the target window before typing starts")
        self._field(grid, 2, 0, "Focus countdown", self.start_delay_spin)

        self.physical_keys_check = QCheckBox("Key holds / rollover")
        self.physical_keys_check.setChecked(True)
        self.physical_keys_check.setToolTip(
            "Use learned key holds; turn off if target app mishandles held keys"
        )
        self._field(grid, 2, 1, "Physical keys", self.physical_keys_check)
        layout.addLayout(grid)

        for control in (self.wpm_spin, self.error_spin, self.rhythm_spin, self.pause_spin, self.start_delay_spin):
            control.valueChanged.connect(self._save_settings)
        self.physical_keys_check.toggled.connect(self._save_settings)

        self.type_button = QPushButton("Start typing")
        self.type_button.setMinimumHeight(44)
        self.type_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.type_button.clicked.connect(self._start_typing)
        layout.addWidget(self.type_button)

        self.status_label = QLabel("Ready to type")
        self._status_effect = QGraphicsOpacityEffect(self.status_label)
        self.status_label.setGraphicsEffect(self._status_effect)
        self._status_animation = QPropertyAnimation(self._status_effect, b"opacity", self)
        self._status_animation.setDuration(180)
        self._status_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)
        self._refresh_model_status()

    @staticmethod
    def _field(grid: QGridLayout, row: int, column: int, title: str, control: QWidget) -> None:
        label = QLabel(title)
        grid.addWidget(label, row * 2, column)
        grid.addWidget(control, row * 2 + 1, column)

    def _refresh_model_status(self) -> None:
        path = self.MODEL_PATH
        model_ready = path.exists()
        self.pause_spin.setEnabled(model_ready)
        self.physical_keys_check.setEnabled(model_ready)
        if path.exists():
            self.model_label.setText("v1")
            self.type_button.setEnabled(True)
        else:
            self.model_label.setText("MODEL MISSING")
            self.type_button.setEnabled(False)

    def _load_settings(self) -> None:
        self._source_name = self._settings.value("source_name", "Draft", type=str)
        values = (
            (self.text_edit, "text", "", str),
            (self.wpm_spin, "wpm", 75, int),
            (self.error_spin, "error_rate_percent", 3.5, float),
            (self.rhythm_spin, "rhythm_variation_percent", 100, int),
            (self.pause_spin, "pause_tendency_percent", 100, int),
            (self.start_delay_spin, "start_delay_ms", 1800, int),
        )
        for widget, key, default, value_type in values:
            blocker = QSignalBlocker(widget)
            value = self._settings.value(key, default, type=value_type)
            if widget is self.text_edit:
                widget.setPlainText(value)
            else:
                widget.setValue(value)
            del blocker
        blocker = QSignalBlocker(self.physical_keys_check)
        self.physical_keys_check.setChecked(self._settings.value("physical_keys", True, type=bool))
        del blocker
        self._update_source_label()

    def _save_settings(self) -> None:
        self._settings.setValue("text", self.text_edit.toPlainText())
        self._settings.setValue("source_name", self._source_name)
        self._settings.setValue("wpm", self.wpm_spin.value())
        self._settings.setValue("error_rate_percent", self.error_spin.value())
        self._settings.setValue("rhythm_variation_percent", self.rhythm_spin.value())
        self._settings.setValue("pause_tendency_percent", self.pause_spin.value())
        self._settings.setValue("physical_keys", self.physical_keys_check.isChecked())
        self._settings.setValue("start_delay_ms", self.start_delay_spin.value())

    def _on_text_changed(self) -> None:
        self._update_source_label()
        self._save_settings()

    def _update_source_label(self) -> None:
        count = len(self.text_edit.toPlainText())
        name = self._source_name
        self.source_label.setText(f"{name} · {count:,} characters")
        self.source_label.setToolTip(name)

    def _choose_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load text", "", "Text files (*.txt *.md *.log *.csv *.json);;All files (*)"
        )
        if selected:
            self._load_file(Path(selected))

    def _load_file(self, path: Path) -> bool:
        try:
            if not path.is_file():
                raise ValueError("Selected path is not a file.")
            if path.stat().st_size > self.MAX_FILE_BYTES:
                raise ValueError("File exceeds 2 MB limit. Choose a shorter text file.")
            content = path.read_text(encoding="utf-8-sig")
            if "\x00" in content:
                raise ValueError("File contains binary data. Choose a UTF-8 text file.")
        except (OSError, UnicodeError, ValueError) as error:
            QMessageBox.warning(self, "Could not load file", str(error))
            return False
        self._source_name = path.name
        self.text_edit.setPlainText(content)
        self._update_source_label()
        self._save_settings()
        self._set_status(f"Loaded {path.name}")
        return True

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        if reduced_motion_enabled():
            return
        self._status_animation.stop()
        self._status_animation.setStartValue(0.55)
        self._status_animation.setEndValue(1.0)
        self._status_animation.start()

    def _start_typing(self) -> None:
        if self._worker is not None:
            return
        text = self.text_edit.toPlainText()
        if not text:
            QMessageBox.warning(self, "pytyper", "Enter text or load a file before typing.")
            return
        if not self.MODEL_PATH.exists():
            QMessageBox.critical(self, "pytyper", "Trained model file is missing.")
            self._refresh_model_status()
            return

        behavior = TypingControls(
            wpm=self.wpm_spin.value(),
            error_rate=self.error_spin.value() / 100.0,
            rhythm_variation=self.rhythm_spin.value() / 100.0,
            pause_tendency=self.pause_spin.value() / 100.0,
            physical_keys=self.physical_keys_check.isChecked(),
        )
        config = TypingConfig(text, self.start_delay_spin.value(), behavior, str(self.MODEL_PATH))
        self._thread = QThread(self)
        self._worker = TypingWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status_changed.connect(self._set_status)
        self._worker.status_changed.connect(self.progress_overlay.set_status)
        self._worker.progress_changed.connect(self.progress_overlay.update_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self.type_button.setEnabled(False)
        self.hide()
        self.progress_overlay.show_running()
        self._thread.start()

    def _stop_typing(self) -> None:
        if self._worker is not None:
            self.progress_overlay.set_status("Stopping typing…")
            self._worker.request_stop()

    def _pause_typing(self) -> None:
        if self._worker is not None:
            self._worker.request_pause()

    def _resume_typing(self) -> None:
        if self._worker is not None:
            self._worker.request_resume()

    def _on_worker_finished(self, _canceled: bool, message: str) -> None:
        self._set_status(message)
        self._save_settings()
        self.progress_overlay.show_finished(message)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._refresh_model_status()

    def _show_main(self) -> None:
        self.progress_overlay.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self._save_settings()
        if self._worker is not None:
            self._worker.request_stop()
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(1500)
        self.progress_overlay.close()
        super().closeEvent(event)
