"""Application entry point."""
from __future__ import annotations

import logging
import sys


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow
    from .theme import apply_theme

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("CVFlow")
    app.setOrganizationName("cvflow")
    apply_theme(app)
    path = next((a for a in argv[1:] if a.endswith(".json")), None)
    win = MainWindow(path)
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
