from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

router = APIRouter()
model = tf.keras.models.load_model("models/transfer_image_model.keras")

@router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    
    image = Image.open(io.BytesIO(await file.read())).resize((128, 128)).convert("RGB")
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "chien" if prediction > 0.5 else "chat"
    return JSONResponse({"prediction": label, "confidence": float(prediction)})