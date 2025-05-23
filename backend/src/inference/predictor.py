import numpy as np
from keras.models import load_model
from src.audio_model.train import extract_mfcc
from src.inference.utils import preprocess_image
from config.logger_config import configure_logger

logger = configure_logger()

def predict_multimodal(image_path, audio_path, model_path='models/multimodal_model.keras'):
    model = load_model(model_path)

    img = preprocess_image(image_path)
    mfcc = extract_mfcc(audio_path)

    if img is None or mfcc is None:
        logger.error("Erreur de prétraitement")
        return None

    img = np.expand_dims(img, axis=0)             # (1, 128, 128, 3)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]      # (1, n_mfcc, time, 1)

    pred = model.predict([img, mfcc])[0][0]
    label = 'dog' if pred > 0.5 else 'cat'

    logger.info(f"[Multimodal] → Score: {pred:.4f} → Label: {label}")
    return {'score': float(pred), 'label': label}