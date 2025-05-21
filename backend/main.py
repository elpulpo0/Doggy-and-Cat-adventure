from config.logger_config import configure_logger
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Configuration du logger
logger = configure_logger()

app = FastAPI()
model = tf.keras.models.load_model('models/cnn_image_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "chien" if prediction > 0.5 else "chat"
    return JSONResponse({"prediction": label, "confidence": float(prediction)})