import logging
from typing import Optional

def setup_logger(log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Setup logging configuration with ANSI colors for console output."""

    logger = logging.getLogger("ScriptRunner")
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    class CustomFormatter(logging.Formatter):      
        COLORS = {
            'SUCCESS': '\033[92m',  # Green
            'FAIL': '\033[91m',     # Red
            'WARN': '\033[93m',     # Yellow
            'INFO': '\033[94m',     # Blue
            'RESET': '\033[0m',     # Reset to default
        }

        def format(self, record):
            msg = record.msg

            if '[SUCCESS]' in msg:
                color = self.COLORS['SUCCESS']
            elif '[FAIL]' in msg:
                color = self.COLORS['FAIL']
            elif '[WARN]' in msg:
                color = self.COLORS['WARN']
            elif '[INFO]' in msg:
                color = self.COLORS['INFO']
            else:
                color = self.COLORS['RESET']

            reset = self.COLORS['RESET']
            self._style._fmt = f"{color}%(asctime)s - %(message)s{reset}"
            return super().format(record)


    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger

