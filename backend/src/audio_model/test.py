import os
import numpy as np
from keras.models import load_model
from src.audio_model.train import extract_mfcc
from config.logger_config import configure_logger

logger = configure_logger()

def predict_audio(model_path='models/cnn_audio_model.keras', test_dir='data/audio/test', batch_size=16):
    model = load_model(model_path)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    audio_results = []
    all_mfccs = []
    all_fnames = []

    for label_folder in ['cats', 'dogs']:
        folder_path = os.path.join(test_dir, label_folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

        for fname in files:
            path = os.path.join(folder_path, fname)
            mfcc = extract_mfcc(path)
            if mfcc is None:
                continue
            mfcc = mfcc[np.newaxis, ..., np.newaxis]  # shape (1, time, features, 1)
            all_mfccs.append(mfcc)
            all_fnames.append(fname)

    if not all_mfccs:
        logger.warning("Pas de MFCC valide extrait.")
        return []

    X = np.vstack(all_mfccs)
    num_batches = int(np.ceil(len(X) / batch_size))

    logger.info(f"Traitement de {len(all_fnames)} sons séparés en {num_batches} batches.")

    preds = model.predict(X, verbose=1, batch_size=batch_size)

    for idx, pred in enumerate(preds):
        score = float(pred[0])
        predicted_label = 'dog' if score > 0.5 else 'cat'
        audio_results.append({
            'file': all_fnames[idx],
            'score': round(score, 4),
            'predicted': predicted_label,
        })

    logger.info(f"✅ Prédiction terminée: {len(audio_results)} fichiers traités.")
    return audio_results
