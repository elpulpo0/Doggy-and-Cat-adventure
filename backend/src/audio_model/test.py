import os
import numpy as np
from keras.models import load_model
from src.audio_model.train import extract_mfcc
from config.logger_config import configure_logger

logger = configure_logger()

def predict_audio(model_path='models/cnn_audio_model.keras', test_dir='data/audio/test', batch_size=16, model_type='simple'):
    """
    Prédit les classes audio à partir des fichiers wav dans test_dir,
    avec un modèle Keras chargé depuis model_path.

    Args:
        model_path (str): chemin vers le fichier du modèle Keras (.keras)
        test_dir (str): dossier contenant sous-dossiers par label (ex: 'cats', 'dogs')
        batch_size (int): taille des batches pour la prédiction
        model_type (str): 'simple', 'complex' ou 'transfer' selon le modèle pour adapter l'entrée

    Returns:
        List[dict]: liste des résultats {'file': nom_fichier, 'score': score, 'predicted': label}
    """

    model = load_model(model_path)
    audio_results = []
    all_mfccs = []
    all_fnames = []

    for label_folder in ['cats', 'dogs']:
        folder_path = os.path.join(test_dir, label_folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Dossier non trouvé: {folder_path}")
            continue
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

        for fname in files:
            path = os.path.join(folder_path, fname)
            mfcc = extract_mfcc(path)
            if mfcc is None:
                logger.warning(f"MFCC invalide pour {fname}, fichier ignoré.")
                continue

            # Selon le modèle, ajuste la forme des MFCCs
            if model_type in ['simple', 'complex']:
                # Forme attendue : (time, features, 1)
                mfcc = mfcc[..., np.newaxis]
                mfcc = mfcc[np.newaxis, ...]  # (1, time, features, 1)

            elif model_type == 'transfer':
                # Le modèle transfer attend 3 canaux en entrée (ex: CNN d'images RGB)
                # Donc on répète le canal unique 3 fois
                mfcc = mfcc[..., np.newaxis]  # (time, features, 1)
                mfcc = np.repeat(mfcc, 3, axis=-1)  # (time, features, 3)
                mfcc = mfcc[np.newaxis, ...]  # (1, time, features, 3)

            else:
                logger.error(f"Type de modèle inconnu: {model_type}")
                return []

            all_mfccs.append(mfcc)
            all_fnames.append(fname)

    if not all_mfccs:
        logger.warning("Pas de MFCC valide extrait, arrêt de la prédiction.")
        return []

    try:
        X = np.vstack(all_mfccs)
    except Exception as e:
        logger.error(f"Erreur lors de la concaténation des MFCCs: {e}")
        return []

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
