import os
import numpy as np
from keras.models import load_model
from src.audio_model.train import extract_mfcc
from config.logger_config import configure_logger

logger = configure_logger()

def predict_audio(model_path='models/cnn_audio_model.keras', test_dir='data/audio/test'):
    model = load_model(model_path)

    # Recompiler si nécessaire (surtout pour évaluer)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    audio_results = []

    for label_folder in ['cats', 'dogs']:
        folder_path = os.path.join(test_dir, label_folder)
        for fname in os.listdir(folder_path):
            if fname.endswith('.wav'):
                path = os.path.join(folder_path, fname)
                mfcc = extract_mfcc(path)
                if mfcc is None:
                    continue
                mfcc = mfcc[np.newaxis, ..., np.newaxis]
                pred = model.predict(mfcc)[0][0]
                predicted_label = 'dog' if pred > 0.5 else 'cat'
                logger.info(f"{fname} → Score: {pred:.4f} → Predicted: {predicted_label}")
                
                audio_results.append({
                    'file': fname,
                    'score': round(float(pred), 4),
                    'predicted': predicted_label,
                })

    return audio_results
