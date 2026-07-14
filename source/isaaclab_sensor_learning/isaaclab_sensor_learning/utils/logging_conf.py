#!/usr/bin/env python3
import os
import logging
import logging.config

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PKG_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "isaaclab_sensor_learning.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        },
        "verbose": {
            "format": "[%(asctime)s] %(levelname)s %(name)s " "(%(filename)s:%(lineno)d): %(message)s",
        },
        "colored": {
            "()": lambda: ColorFormatter(),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level": os.getenv("LOG_CONSOLE_LEVEL", "DEBUG"),
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "verbose",
            "level": os.getenv("ISAACLAB_SENSOR_LEARNING_LOG_LEVEL", "DEBUG"),
            "filename": os.getenv("ISAACLAB_SENSOR_LEARNING_LOG_FILE", LOG_FILE),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "OpenGL.GL.shaders": {"level": "WARNING"},
        "OpenGL.acceleratesupport": {"level": "WARNING"},
        "trimesh": {"level": "WARNING"},
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    light_green = "\x1b[92;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: light_green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    return


setup_logging()