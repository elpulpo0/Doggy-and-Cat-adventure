"""
Flow Prefect : surveillance + (ré)entraînement des modèles Vision & Multimodal
──────────────────────────────────────────────────────────────────────────────
  1. Récupère les accuracies exposées par l’API FastAPI
  2. Si l’une < THRESH → retrain la branche concernée
  3. Met à jour les gauges Prometheus + message Discord
  4. Planifié toutes les 2 minutes via IntervalSchedule
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import os, re, httpx
from typing import Dict, Optional

from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
from prometheus_client import Gauge

# ───── Constantes env ──────────────────────────────────────────
API_ML   = os.getenv("ML_API_URL", "http://fastapi-app:8000")
DISCORD  = os.getenv("DISCORD_WEBHOOK_URL")
THRESH   = float(os.getenv("DC_THRESH", "0.90"))

VISION_ACC = Gauge("model_vision_accuracy", "Accuracy vision model")
MULTI_ACC  = Gauge("model_multi_accuracy",  "Accuracy multimodal model")

MODEL_DIR = Path("/app/models")

# ───── Helper : récupérer la métrique --------------------------
PROM_RE = re.compile(r"^model_(\w+)_accuracy\s+([\d.]+)$", re.MULTILINE)

def fetch_metric(name: str) -> Optional[float]:
    """
    Essaye d'abord /metrics/<name>_acc (JSON), sinon parse le texte Prometheus.
    Retourne None si la métrique n'existe pas encore.
    """
    try:
        r = httpx.get(f"{API_ML}/metrics/{name}_acc", timeout=5)
        if r.status_code == 200 and "value" in r.json():
            return float(r.json()["value"])
    except Exception:
        pass

    try:
        text = httpx.get(f"{API_ML}/metrics", timeout=5).text
        m = re.search(rf"model_{name}_accuracy\s+([\d.]+)", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    return None  # non disponible

# ───── Tasks Prefect ───────────────────────────────────────────
@task
def evaluate_models() -> Dict[str, float]:
    """Renvoie les accuracies et met à jour les gauges (si valeur connue)."""
    logger = get_run_logger()

    img_score   = fetch_metric("vision")
    multi_score = fetch_metric("multi")

    if img_score is not None:
        VISION_ACC.set(img_score)
    if multi_score is not None:
        MULTI_ACC.set(multi_score)

    logger.info(f"Vision={img_score}, Multi={multi_score}")
    return {"vision": img_score, "multi": multi_score}

@task
def retrain(branch: str):
    logger = get_run_logger()
    if branch == "vision":
        from src.image_model.train import train_image_model
        train_image_model(
            model_type="transfer",
            data_dir="data/images/train",
            model_path=MODEL_DIR / "transfer_cnn_image_model.keras",
            use_wandb=True,
        )
    elif branch == "multi":
        from src.multimodal.train import train_multimodal_yamnet
        train_multimodal_yamnet()
    logger.info(f"✅ Retrain {branch} terminé")

@task
def send_discord(msg: str):
    if DISCORD:
        try:
            httpx.post(DISCORD, json={"content": msg}, timeout=10)
        except Exception:
            pass

# ───── Flow principal ──────────────────────────────────────────
@flow(name="monitor-dogcat")
def monitor_dogcat(threshold: float = THRESH) -> None:
    scores    = evaluate_models()
    retrained = []

    for branch, score in scores.items():
        # Si métrique absente → force un 1ᵉʳ retrain pour amorcer la gauge
        if score is None or score < threshold:
            retrain(branch)
            retrained.append((branch, score))

    if retrained:
        lines = "\n".join([f"• {b}: {s if s is not None else 'N/A'}" for b, s in retrained])
        send_discord(f"🔁 Retrain déclenché :\n{lines}")
    else:
        send_discord(f"✅ Scores OK {scores}")

# ───── Déploiement auto (chaque 2 min) ─────────────────────────
def create_deployment() -> None:
    Deployment.build_from_flow(
        flow           = monitor_dogcat,
        name           = "dogcat-monitor",
        work_pool_name = "default-agent-pool",
        schedule       = IntervalSchedule(interval=timedelta(minutes=2)),
    ).apply()

if __name__ == "__main__":
    create_deployment()
    monitor_dogcat()