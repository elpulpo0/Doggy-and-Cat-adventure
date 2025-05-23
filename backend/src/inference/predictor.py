import numpy as np
import tensorflow as tf
from keras.models import load_model
from src.audio_model.train import extract_mfcc
from src.inference.utils import preprocess_image
from config.logger_config import configure_logger

logger = configure_logger()

class MultimodalPredictor:
    def __init__(self, model_path):
        logger.info(f"Chargement du modèle depuis {model_path}...")
        self.model = load_model(model_path)
        logger.info("Modèle chargé avec succès.")

    def preprocess(self, image_path, audio_path):
        # Prétraitement image
        img = preprocess_image(image_path)
        if img is None:
            raise ValueError(f"Erreur lors du chargement ou prétraitement de l'image: {image_path}")

        # Prétraitement audio
        mfcc = extract_mfcc(audio_path)
        if mfcc is None:
            raise ValueError(f"Erreur lors du chargement ou prétraitement de l'audio: {audio_path}")

        # Ajuster la largeur à 173 frames (padding si besoin)
        if mfcc.shape[1] > 173:
            mfcc = mfcc[:, :173]
        elif mfcc.shape[1] < 173:
            pad_width = 173 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        # Normalisation standard
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std = np.std(mfcc, axis=1, keepdims=True)
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)

        # Ajouter 3 canaux (pour correspondre au modèle)
        mfcc = np.repeat(mfcc[..., np.newaxis], 3, axis=-1)

        # Ajouter une dimension batch
        img = np.expand_dims(img, axis=0)
        mfcc = np.expand_dims(mfcc, axis=0)

        return img, mfcc


    def predict(self, image_path, audio_path):
        img, mfcc = self.preprocess(image_path, audio_path)

        pred = self.model.predict([img, mfcc])

        confidence = float(pred[0][0])
        class_idx = int(confidence > 0.5)
        if class_idx == 1:
            label = "dogs"
            confidence = confidence
        else:
            label = "cats"
            confidence = 1 - confidence

        return class_idx, confidence, label
