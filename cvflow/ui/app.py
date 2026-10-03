"""Application entry point."""
from __future__ import annotations

import logging
import sys


def install_qt_translations(app) -> None:
    """加载 Qt 自带的中文翻译，使 OK/Cancel 等标准按钮和对话框显示中文。"""
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for name in ("qtbase", "qt"):
        tr = QTranslator(app)
        if tr.load(QLocale("zh_CN"), name, "_", path):
            app.installTranslator(tr)


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
    install_qt_translations(app)
    path = next((a for a in argv[1:] if a.endswith(".json")), None)
    win = MainWindow(path)
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
