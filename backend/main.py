from config.logger_config import configure_logger
from fastapi import FastAPI
from app.api import image
from app.api import audio_yamnet

# Configuration du logger
logger = configure_logger()

app = FastAPI()
app.include_router(image.router)
app.include_router(audio_yamnet.router, prefix="/predict")