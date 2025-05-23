from config.logger_config import configure_logger
from fastapi import FastAPI
from app.api import image

# Configuration du logger
logger = configure_logger()

app = FastAPI()
app.include_router(image.router)