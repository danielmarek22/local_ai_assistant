import logging
import logging.config
import sys
from pathlib import Path


_LOGGING_CONFIGURED = False


def setup_logging(level=logging.INFO, log_dir: str = "logs"):
    global _LOGGING_CONFIGURED
    
    if _LOGGING_CONFIGURED:
        logging.getLogger(__name__).debug("Logging already configured, skipping")
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "assistant.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "maxBytes": 10_000_000,  # 10 MB
                "backupCount": 5,
                "encoding": "utf-8",
                "formatter": "default",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(logging_config)

    root = logging.getLogger()
    root.info("Logging initialized")
    root.info("Log level set to %s", logging.getLevelName(level))
    root.info("Log file: %s", log_file.resolve())

    _LOGGING_CONFIGURED = True
