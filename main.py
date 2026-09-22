"""pytyper application entrypoint."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from app.window import PytyperWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("pytyper")
    app.setApplicationName("pytyper")
    window = PytyperWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
