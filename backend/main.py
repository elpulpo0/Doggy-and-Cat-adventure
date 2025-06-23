from pathlib import Path
import logging

# 1. Filtre des requêtes /metrics et /health dans uvicorn.access
class FilterNoise(logging.Filter):
    """Ignore les logs GET /metrics et /health de l'access-logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Conserve le log uniquement si aucun des deux chemins n'est présent
        return ("/metrics" not in msg) and ("/health" not in msg)

uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.addFilter(FilterNoise())

# 2. Application FastAPI + instrumentation Prometheus
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# Routes « produit »
from app.api.image import router as image_router
from app.api.audio_yamnet import router as audio_router
from app.api.predict_multimodal import router as multi_router

# Routes MLOps / monitoring
from app.api.routes_monitoring import router as monitor_router

app = FastAPI(
    title="Doggy & Cat Adventure API",
    version="1.0.0",
)

# Expose /metrics
Instrumentator().instrument(app).expose(app)

# Routes prédiction
app.include_router(image_router, prefix="/predict")
app.include_router(audio_router, prefix="/predict")
app.include_router(multi_router, prefix="/predict")

# Routes monitoring
app.include_router(monitor_router)

@app.get("/")
def root():
    return {"msg": "Doggy & Cat API — OK"}