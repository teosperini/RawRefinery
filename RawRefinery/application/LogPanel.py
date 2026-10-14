import html
import logging
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from platformdirs import user_log_dir
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtCore import Qt

COLORS = {
    "info": "#d0d0d0",
    "success": "#7ee787",
    "error": "#ff7b72",
    "history": "#6e7681",
}


class LogPanel(QPlainTextEdit):
    """Black, terminal-style event log.

    Messages are timestamped, shown in a scrollable view and appended to a log file,
    so the history of previous sessions is shown again on start-up and never lost.
    """

    def __init__(self, parent=None, history_lines=300):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setMinimumHeight(70)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.setStyleSheet(
            "QPlainTextEdit { background-color: #0c0c0c; color: #d0d0d0;"
            " border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px; }"
        )

        self._file_logger = self._make_file_logger()
        self._load_history(history_lines)
        self._append_html("── new session ──", "history")

    # File logging
    def _log_path(self) -> Path:
        return Path(user_log_dir("RawRefinery")) / "rawrefinery.log"

    def _make_file_logger(self):
        try:
            path = self._log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger("rawrefinery.ui")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            return logger
        except OSError:
            return None  # logging to file is best effort

    def _load_history(self, n):
        try:
            with open(self._log_path(), encoding="utf-8") as f:
                for line in deque(f, maxlen=n):
                    self._append_html(line.rstrip("\n"), "history")
        except OSError:
            pass

    # Display
    def _append_html(self, text, level):
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2   # don't yank the view while reading old lines
        color = COLORS.get(level, COLORS["info"])
        self.appendHtml(f'<span style="color:{color}; white-space:pre-wrap;">{html.escape(text)}</span>')
        if at_bottom:
            bar.setValue(bar.maximum())

    def log(self, message, level=None):
        message = str(message).strip()
        if not message:
            return
        if level is None:
            low = message.lower()
            if low.startswith("error") or "failed" in low:
                level = "error"
            elif low.startswith("done"):
                level = "success"
            else:
                level = "info"
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
        self._append_html(line, level)
        if self._file_logger is not None:
            self._file_logger.info(line)

    # Same name as QLabel.setText so existing status updates become log entries.
    def setText(self, message):
        self.log(message)
