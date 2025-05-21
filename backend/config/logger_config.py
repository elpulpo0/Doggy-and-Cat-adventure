from loguru import logger
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def configure_logger():
    logger.remove()

    # Créer un dossier de logs s'il n'existe pas
    log_dir = BASE_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Format de log avec une coloration automatique des niveaux grâce à la
    # balise <level>
    log_format = (
        "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
        "<blue>{name}</blue> | "
        "<level>{level}</level> | "
        "<magenta>{message}</magenta>"
    )

    # Console
    logger.add(sys.stderr, level="DEBUG", format=log_format)

    # Fichier général (tous les logs)
    logger.add(
        f"{log_dir}/app.log",
        rotation="1 week",
        retention="1 month",
        level="INFO",
        format=log_format,
    )

    # Fichier uniquement pour ERROR
    logger.add(
        f"{log_dir}/error.log",
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR",
        rotation="500 KB",
        retention="10 days",
        format=log_format,
    )

    # Fichier uniquement pour DEBUG
    logger.add(
        f"{log_dir}/debug.log",
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
        rotation="500 KB",
        retention="10 days",
        format=log_format,
    )

    return logger
