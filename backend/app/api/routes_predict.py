from fastapi import APIRouter, UploadFile
from . import image, audio_yamnet, predict_multimodal  # import relatifs

router = APIRouter()

@router.post("/predict/image")
async def predict_image(file: UploadFile):
    return image.predict_image(file)

@router.post("/predict/audio-yamnet")
async def predict_audio(file: UploadFile):
    return audio_yamnet.predict_audio(file)

@router.post("/predict/multimodal")
async def predict_multi(img: UploadFile, audio: UploadFile):
    return predict_multimodal.predict(img, audio)