import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from config.logger_config import configure_logger
from src.inference.utils import load_image_tensor

logger = configure_logger()


def extract_yamnet_embedding(wav_path):
    waveform, sr = librosa.load(wav_path, sr=16000)
    waveform = waveform[:16000 * 10]
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = hub.load("https://tfhub.dev/google/yamnet/1")(waveform_tensor)
    return tf.reduce_mean(embeddings, axis=0).numpy()


def test_multimodal_yamnet():
    model = tf.keras.models.load_model("models/multimodal_yamnet_model.keras")
    img = load_image_tensor("data/images/test/dog.9999.jpg")
    audio = extract_yamnet_embedding("data/audio/test/dog_barking_99.wav")

    img_batch = np.expand_dims(img, axis=0)
    audio_batch = np.expand_dims(audio, axis=0)

    prediction = model.predict([img_batch, audio_batch])[0][0]
    label = "chien" if prediction > 0.5 else "chat"
    logger.info(f"🐾 Prédiction : {label.upper()} ({round(prediction*100, 2)}%)")