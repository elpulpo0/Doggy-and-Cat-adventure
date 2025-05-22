import os
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from config.logger_config import configure_logger

logger = configure_logger()

def predict_on_test_images_batch(model_path, test_dir='data/images/test', threshold=0.5, batch_size=32):
    model = load_model(model_path)

    # Chargement dataset test par batch avec tf.keras.utils.image_dataset_from_directory
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(128, 128),
        batch_size=batch_size,
        label_mode=None,  # Pas besoin des labels pour la prédiction
        shuffle=False
    )

    # Normalisation (comme à l'entraînement)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x: normalization_layer(x))

    # Nombre de batches
    num_batches = tf.data.experimental.cardinality(test_ds).numpy()

    # Liste des noms de fichiers
    filenames = sorted([
        fname for fname in os.listdir(test_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    logger.info(f"Traitement de {len(filenames)} fichiers séparés en {num_batches} batches.")

    # Prédiction
    preds = model.predict(test_ds, verbose=1)
    
    # Résultats
    results = []
    for idx, pred in enumerate(preds):
        score = float(pred[0])
        label = 'dog' if score > threshold else 'cat'
        fname = filenames[idx]
        results.append({'filename': fname, 'score': score, 'label': label})

    logger.info(f"✅ Prédiction terminée: {len(filenames)} fichiers traités.")
    return pd.DataFrame(results)
