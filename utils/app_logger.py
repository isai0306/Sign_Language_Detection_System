"""Application logging setup."""
import logging
import os
from logging.handlers import RotatingFileHandler


def init_app_logging(app_name: str = "signai") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log = logging.getLogger(app_name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    fh = RotatingFileHandler(
        os.path.join("logs", "signai.log"),
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def get_logger(name: str = "signai") -> logging.Logger:
    return logging.getLogger(name)
