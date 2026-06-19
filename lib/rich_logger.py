"""
coalition_logging.py
Rich-coloured, single-source logger for the coalition formation project.
Import   >>> from lib.logging import get_logger, setup_logging
then     >>> logger = get_logger(__name__)
"""

import logging
from pathlib import Path
from typing import Optional
import sys
from rich.console import Console
from rich.logging import RichHandler
import socket


# ── Custom FileHandler that flushes immediately ──────────────────────
class FlushingFileHandler(logging.FileHandler):
    """FileHandler that explicitly flushes after every message."""
    def emit(self, record):
        try:
            super().emit(record)
            # Explicitly flush to ensure messages are written immediately
            if self.stream:
                self.stream.flush()
        except Exception:
            self.handleError(record)


# ── Unicode/Emoji support detection ──────────────────────────────────────
def safe_emoji_text(emoji_text: str, fallback_text: str = None) -> str:
    """
    Safely return emoji text if Unicode is supported, otherwise return empty string.
    """
    try:
        # Test if we can encode the emoji
        emoji_text.encode(sys.stdout.encoding or 'utf-8')
        return emoji_text
    except (UnicodeEncodeError, LookupError):
        # Return empty string if emojis can't be displayed
        return ""


# ── custom SUCCESS level ───────────────────────────────────────────────────
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, msg, args, **kwargs)


logging.Logger.success = _success  # type: ignore

# ── Logging configuration ───────────────────────────────────────────────────
# Default log directory
DEFAULT_LOG_DIR = Path("./logs")

# ── one-time Rich configuration (executes on first import) ────────────────
# Simple console setup with error handling for Unicode
try:
    console = Console(
        file=sys.stdout,
        force_terminal=True,
        legacy_windows=False
    )
except Exception:
    # Fallback to basic console if there are encoding issues
    console = Console(force_terminal=True)

rich_handler = RichHandler(
    console=console,
    markup=True,          # parse [bold green]…[/] etc.
    rich_tracebacks=True,
    show_path=False,      # flip to True if you want file:line info
)


# ── setup_logging function for advanced configuration ─────────────────────
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None,
                  log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging with RichHandler and file handler.

    Args:
        log_level: Logging level (default: "INFO")
        log_file: Optional specific log file path (for run-specific logging)
        log_dir: Optional log directory (default: ./logs)

    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("coalition_formation")
    logger.setLevel(logging.DEBUG)

    # Define formatters
    machine_name = socket.gethostname()
    file_formatter = logging.Formatter(
        f'%(asctime)s - {machine_name} - %(levelname)s - %(message)s'
    )

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add RichHandler for console logging
    rich_handler_copy = RichHandler(
        console=console,
        markup=True,
        rich_tracebacks=True,
        show_path=False,
    )
    rich_handler_copy.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(rich_handler_copy)

    # Add run-specific log file if provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = FlushingFileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# ── public helper ----------------------------------------------------------
def get_logger(log_file: Optional[Path] = None, log_level: str = "INFO") -> logging.Logger:
    """
    Return a Rich-enabled logger.

    Args:
        log_file: Optional log file path (creates scenario-specific log)
        log_level: Logging level (default: "INFO")

    Returns:
        Configured logger instance
    """
    return setup_logging(log_level=log_level, log_file=log_file)
