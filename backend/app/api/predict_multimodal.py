from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import librosa

router = APIRouter()

# Charger le modèle fusionné image + audio
model = tf.keras.models.load_model("models/multimodal_model.keras")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

@router.post("/multimodal")
async def predict_multimodal(image: UploadFile = File(...), audio: UploadFile = File(...)):
    try:
        # Traitement de l'image
        img = Image.open(io.BytesIO(await image.read())).resize((128, 128))
        img_array = np.array(img) / 255.0
        if img_array.ndim == 2:  # grayscale → RGB
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA → RGB
            img_array = img_array[..., :3]
        img_array = img_array[np.newaxis, ...]  # (1, 128, 128, 3)

        # Traitement de l'audio via YAMNet
        audio_bytes = await audio.read()
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        waveform = waveform[:16000 * 10]  # max 10 sec
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform)
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

        # Prédiction
        prediction = model.predict([img_array, mean_embedding])[0][0]
        label = "chien" if prediction > 0.5 else "chat"
        return JSONResponse({"prediction": label, "confidence": float(prediction)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)