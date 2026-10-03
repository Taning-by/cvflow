"""PySide6 desktop application."""


def main(argv=None) -> int:
    from .app import main as _main
    return _main(argv)
