from loguru import logger
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def log_format(record):
    my_name = record["extra"].get("my_name", record["name"])
    custom = "Custom logs by El Pulpo"
    return (
        f"<green>{custom}</green> | "
        f"<cyan>{record['time']:YYYY-MM-DD HH:mm:ss}</cyan> | "
        f"<blue>{my_name}</blue> | "
        f"<level>{record['level'].name}</level> | "
        f"<magenta>{record['message']}</magenta>\n"
    )

def configure_logger(my_name=None):
    logger.remove()

    log_dir = BASE_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger.add(sys.stderr, level="DEBUG", format=log_format)
    logger.add(
        f"{log_dir}/app.log",
        rotation="1 week",
        retention="1 month",
        level="INFO",
        format=log_format,
    )
    logger.add(
        f"{log_dir}/error.log",
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR",
        rotation="500 KB",
        retention="10 days",
        format=log_format,
    )
    logger.add(
        f"{log_dir}/debug.log",
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
        rotation="500 KB",
        retention="10 days",
        format=log_format,
    )

    if my_name:
        return logger.bind(my_name=my_name)
    else:
        return logger
