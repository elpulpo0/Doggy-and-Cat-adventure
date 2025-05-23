from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import io

router = APIRouter()

# Charger le classifieur entraîné et le modèle YAMNet
classifier = tf.keras.models.load_model("models/yamnet_audio_classifier.keras")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

@router.post("/audio-yamnet")
async def predict_audio_yamnet(file: UploadFile = File(...)):
    try:
        # Lecture et chargement du fichier audio
        audio_bytes = await file.read()
        waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        waveform = waveform[:16000 * 10]  # max 10 sec
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

        # Extraction des embeddings avec YAMNet
        _, embeddings, _ = yamnet_model(waveform)
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

        # Prédiction avec le classifieur entraîné
        prediction = classifier.predict(mean_embedding)[0][0]
        label = "chien" if prediction > 0.5 else "chat"

        return JSONResponse({"prediction": label, "confidence": float(prediction)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)