# logger.py

import os
import logging
import inspect
from pathlib import Path
from typing import Optional, Union, Literal

_initialized = False
_verbose: Literal["half", "full"] = "full"

class CustomFormatter(logging.Formatter):
    # ANSI codes for colorized console output
    COLORS = {
        'START':   '\033[94m',  # Green
        'SUCCESS': '\033[92m',  # Green
        'FINISH':  '\033[94m',  # Blue
        'ERROR':   '\033[91m',  # Red
        'FAIL':    '\033[91m',  # Red
        'WARN':    '\033[93m',  # Yellow
        'INFO':    '',          # default
        'DEBUG':   '',          # default
    }
    RESET = '\033[0m'


    def format(self, record):
        # figure out which phase tag is in the message
        text = record.getMessage()
        phase = None
        for tag in self.COLORS:
            if f'[{tag}]' in text:
                phase = tag
                break

        color = self.COLORS.get(phase, '')

        if color:
            fmt = f"{color}%(asctime)s - %(message)s{self.RESET}"
        else:
            fmt = "%(asctime)s - %(message)s"

        self._style._fmt = fmt
        return super().format(record)

def setup_logging(
    log_file: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    verbose: Literal["half", "full"] = "full"
) -> None:
    """
    Initialize the root logger. Call this ONCE in your main script.
    :param log_file: if given, all logs will also be written here.
    :param level: logging level, e.g. "DEBUG", "INFO", or logging.DEBUG.
    :param verbose: 
        "full" = "<datetime> - [path] - [category] - [PHASE] - 'message'"
        "half" = "<datetime> - [PHASE] - 'message'"
    """
    global _initialized, _verbose
    if _initialized:
        return

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    if verbose not in ["half", "full"]:
        raise ValueError(f"verbose must be 'half' or 'full', got {verbose}")
    _verbose = verbose

    root = logging.getLogger()
    root.setLevel(level)

    # clear out old handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # console handler with our color formatter
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    root.addHandler(ch)

    # optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        ))
        root.addHandler(fh)

    _initialized = True

class Logger:
    """
    A simple logger that prefixes every message with
      <datetime> - [<script_path>] - [<category>] - [PHASE] - "your message"
    and lets the CustomFormatter color it.
    """
    def __init__(
            self, 
            script_path: Optional[str] = None, 
            category:    Optional[str] = None
            ):
        self._script_path = Path(script_path).with_suffix(".py") if script_path else None
        self._category = category
        self._logger = logging.getLogger()

    def _log(self, level: int, phase: str, msg: str, *args, **kwargs):
        prefix = ""
        if _verbose == "full":
            prefix += f"[{self._script_path.name[:15]:<15}] - " if self._script_path else ""
            prefix += f"[{phase+']':<8.30s} - "
            prefix += f"[{self._category}] - " if self._category else ""
        else:
            prefix += f"[{phase}] - "
        message = prefix + f"{msg}"
        self._logger.log(level, message, *args, **kwargs)

    def start(self, msg: str, *a, **kw):   self._log(logging.INFO,   "START",  msg, *a, **kw)
    def finish(self, msg: str, *a, **kw):  self._log(logging.INFO,   "FINISH", msg, *a, **kw)
    def info(self, msg: str, *a, **kw):    self._log(logging.INFO,   "INFO",   msg, *a, **kw)
    def fail(self, msg: str, *a, **kw):    self._log(logging.INFO,   "FAIL",   msg, *a, **kw)
    def success(self, msg: str, *a, **kw): self._log(logging.INFO,   "SUCCESS",msg, *a, **kw)
    def debug(self, msg: str, *a, **kw):   self._log(logging.DEBUG,  "DEBUG",  msg, *a, **kw)
    def warning(self, msg: str, *a, **kw): self._log(logging.WARNING,"WARN",   msg, *a, **kw)
    def error(self, msg: str, *a, **kw):   self._log(logging.ERROR,  "ERROR",  msg, *a, **kw)

def setup_logger(
        category: Optional[str]    = None,
        script_path: Optional[str] = None
        ) -> Logger:
    """
    Returns a Logger whose category is the caller’s script path.
    Make sure you’ve already called setup_logging().
    """
    if _initialized is False:
        raise RuntimeError("Call setup_logging() before setup_logger()")


    if script_path is None:
        frame = inspect.stack()[1]
        full_path = Path(frame.filename).resolve()
        try:
            script_path = full_path.relative_to(Path.cwd())
        except ValueError:
            script_path = full_path.name


    return Logger(
        script_path=script_path, 
        category=category
        )
