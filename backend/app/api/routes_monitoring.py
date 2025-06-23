"""
Routes MLOps (health, generate, predict, retrain, drift, metrics…)
adaptées de l’ancien IA_monitoring/ml_api/main.py
et enrichies pour exposer les métriques vision / multimodal en JSON.
"""

from datetime import datetime, timezone
import os
import sys

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from prometheus_client import (
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

# ────────────────────────────────────────────────
#  Logging basique (console + fichier)
# ────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    f"{LOG_DIR}/ml_api.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time} | {level} | {message}",
)

# ────────────────────────────────────────────────
#  Prometheus gauges
# ────────────────────────────────────────────────
TRAIN_ACCURACY = Gauge("model_train_accuracy", "Training accuracy of the model")
TEST_ACCURACY  = Gauge("model_test_accuracy",  "Test accuracy of the model")
DATASET_SIZE   = Gauge("dataset_size",         "Number of samples in training dataset")

# ↓ Ajouté pour tes vrais modèles vision / multimodal
VISION_ACC = Gauge("model_vision_accuracy",    "Last accuracy of CNN vision model")
MULTI_ACC  = Gauge("model_multi_accuracy",     "Last accuracy of multimodal model")

# ────────────────────────────────────────────────
#  Sécurité  /retrain   (Bearer token)
# ────────────────────────────────────────────────
bearer_scheme = HTTPBearer()

def token_auth(cred: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if cred.credentials != os.getenv("API_TOKEN", ""):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing bearer token",
        )

# ────────────────────────────────────────────────
#  Import managers
# ────────────────────────────────────────────────
from app.api.ml_models import MLModelManager          # noqa
from app.api.database  import DatasetManager          # noqa

dataset_manager   = DatasetManager()
model_manager     = MLModelManager()
generation_counter = 0

# ────────────────────────────────────────────────
#  Pydantic models
# ────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_used: str

class GenerateResponse(BaseModel):
    message: str
    generation_number: int
    samples_generated: int

class RetrainResponse(BaseModel):
    message: str
    train_accuracy: float
    test_accuracy: float
    mlflow_run_id: str

# ────────────────────────────────────────────────
#  APIRouter
# ────────────────────────────────────────────────
router = APIRouter(tags=["monitoring"])

# -------------------- Health ------------------- #
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="OK", timestamp=datetime.now(timezone.utc).isoformat())

# -------------- Generate dataset --------------- #
@router.post("/generate", response_model=GenerateResponse)
def generate_dataset(n_samples: int = 1000) -> GenerateResponse:
    global generation_counter
    generation_counter += 1
    X, y = model_manager.generate_dataset(n_samples)
    dataset_manager.save_dataset(X, y, generation_counter)
    return GenerateResponse(
        message="Dataset généré avec succès",
        generation_number=generation_counter,
        samples_generated=n_samples,
    )

# ------------------- Predict ------------------- #
@router.get("/predict", response_model=PredictionResponse)
def predict() -> PredictionResponse:
    X, _ = dataset_manager.get_latest_dataset()
    if X is None:
        raise HTTPException(404, "Aucun dataset trouvé")
    try:
        pred = model_manager.predict(X.iloc[:1].values)[0]
        conf = (
            float(max(model_manager.model.predict_proba(X.iloc[:1].values)[0]))
            if hasattr(model_manager.model, "predict_proba")
            else 0.5
        )
        return PredictionResponse(
            prediction=int(pred),
            confidence=conf,
            model_used=model_manager.model_path,
        )
    except Exception as e:
        raise HTTPException(500, f"Erreur de prédiction: {e}")

# ------------------- Retrain ------------------- #
@router.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(token_auth)])
def retrain_model() -> RetrainResponse:
    X, y = dataset_manager.get_latest_dataset()
    if X is None:
        raise HTTPException(404, "Aucun dataset pour l'entraînement")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5555"))
    mlflow.set_experiment("ml-api-retraining")

    with mlflow.start_run() as run:
        metrics = model_manager.train_model(X.values, y.values)

        TRAIN_ACCURACY.set(metrics["train_accuracy"])
        TEST_ACCURACY.set(metrics["test_accuracy"])
        DATASET_SIZE.set(len(X))

        mlflow.log_params(
            {"n_samples": len(X), "n_features": X.shape[1], "algorithm": "LogisticRegression"}
        )
        mlflow.log_metrics(
            {"train_accuracy": metrics["train_accuracy"], "test_accuracy": metrics["test_accuracy"]}
        )
        mlflow.sklearn.log_model(model_manager.model, "model")

        return RetrainResponse(
            message="Modèle réentraîné avec succès",
            train_accuracy=metrics["train_accuracy"],
            test_accuracy=metrics["test_accuracy"],
            mlflow_run_id=run.info.run_id,
        )

# -------------------- Drift -------------------- #
@router.get("/drift")
def drift_check():
    return {"drift_score": model_manager.compute_drift()}

# --------------- MLflow status ---------------- #
@router.get("/mlflow-status")
def check_mlflow_status():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5555")
    try:
        mlflow.set_tracking_uri(uri)
        return {"mlflow_uri": uri, "status": "connected", "experiments_count": len(mlflow.search_experiments())}
    except Exception as e:
        return {"mlflow_uri": uri, "status": "error", "error": str(e)}

# ------------------- Metrics ------------------ #
@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------- JSON helpers pour monitor_dogcat ---- #
@router.get("/metrics/vision_acc")
def get_vision_acc():
    return {"value": VISION_ACC._value.get()}

@router.get("/metrics/multi_acc")
def get_multi_acc():
    return {"value": MULTI_ACC._value.get()}