"""
Flow Prefect : surveillance + (ré)entraînement Vision & Multimodal
──────────────────────────────────────────────────────────────────
  1. Récupère les accuracies exposées par l’API FastAPI
  2. Si l’une < THRESH → retrain la branche concernée
  3. Met à jour les gauges Prometheus + Discord
  4. Planifié toutes les 2 min via IntervalSchedule
"""

from __future__ import annotations

import os, re, httpx
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
from prometheus_client import Gauge

# ────── Config env ──────────────────────────────────────────────
API_ML   = os.getenv("ML_API_URL", "http://fastapi-app:8000")
DISCORD  = os.getenv("DISCORD_WEBHOOK_URL")
THRESH   = float(os.getenv("DC_THRESH", "0.90"))

BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
IMG_TRAIN   = DATA_DIR / "images" / "train"
AUDIO_TRAIN = DATA_DIR / "audio"  / "train"
MODEL_DIR   = BASE_DIR / "models"

VISION_ACC = Gauge("model_vision_accuracy", "Accuracy vision model")
MULTI_ACC  = Gauge("model_multi_accuracy",  "Accuracy multimodal model")

# ────── Helper métriques ───────────────────────────────────────
def fetch_metric(name: str) -> Optional[float]:
    """Tente l’endpoint JSON, puis le texte Prometheus. Retourne None si absent."""
    try:
        r = httpx.get(f"{API_ML}/metrics/{name}_acc", timeout=5)
        if r.status_code == 200 and "value" in r.json():
            return float(r.json()["value"])
    except Exception:
        pass

    try:
        txt = httpx.get(f"{API_ML}/metrics", timeout=5).text
        m = re.search(rf"model_{name}_accuracy\s+([\d.]+)", txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None

# ────── Tasks ──────────────────────────────────────────────────
@task
def evaluate_models() -> Dict[str, float]:
    """Met à jour les gauges et renvoie {'vision': acc, 'multi': acc}."""
    logger = get_run_logger()
    img, multi = fetch_metric("vision"), fetch_metric("multi")

    if img is not None:   VISION_ACC.set(img)
    if multi is not None: MULTI_ACC.set(multi)

    logger.info(f"Accuracies — vision={img} | multi={multi}")
    return {"vision": img, "multi": multi}

@task
def retrain(branch: str):
    """Réentraîne la branche voulue sans W&B."""
    logger = get_run_logger()

    try:
        if branch == "vision":
            from src.image_model.train import train_image_model
            train_image_model(
                model_type  = "transfer",
                data_dir    = str(IMG_TRAIN),
                model_path  = str(MODEL_DIR / "transfer_cnn_image_model.keras"),
                use_wandb   = False,        # ← désactivation explicite
            )

        elif branch == "multi":
            from src.multimodal.train import train_multimodal_yamnet
            train_multimodal_yamnet(
                img_folder   = str(IMG_TRAIN),
                audio_folder = str(AUDIO_TRAIN),
                use_wandb    = False,       # ← idem
            )
        logger.info(f"✅ Retrain '{branch}' terminé")

    except ModuleNotFoundError as e:
        logger.error(f"Branche '{branch}' impossible à entraîner ({e})")

@task
def send_discord(msg: str):
    if DISCORD:
        try:
            httpx.post(DISCORD, json={"content": msg}, timeout=10)
        except Exception:
            pass

# ────── Flow principal ─────────────────────────────────────────
@flow(name="monitor-dogcat")
def monitor_dogcat(threshold: float = THRESH):
    scores = evaluate_models()
    retrained = []

    for branch, score in scores.items():
        if score is None or score < threshold:
            retrain(branch)
            retrained.append((branch, score))

    if retrained:
        lines = "\n".join(f"• {b}: {s if s is not None else 'N/A'}" for b, s in retrained)
        send_discord(f"🔁 Retrain déclenché :\n{lines}")
    else:
        send_discord(f"✅ Scores OK {scores}")

# ────── Déploiement auto (2 min) ───────────────────────────────
def create_deployment():
    Deployment.build_from_flow(
        flow           = monitor_dogcat,
        name           = "dogcat-monitor",
        work_pool_name = "default-agent-pool",
        schedule       = IntervalSchedule(interval=timedelta(minutes=2)),
    ).apply()

if __name__ == "__main__":
    create_deployment()
    monitor_dogcat()