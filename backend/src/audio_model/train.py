import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.audio_model.models import (
    build_simple_audio_cnn,
    build_complex_audio_cnn,
    build_transfer_audio_model
)
from config.logger_config import configure_logger
import datetime
import wandb
from wandb.integration.keras import WandbMetricsLogger

logger = configure_logger()

def extract_mfcc(file_path, max_pad_len=173, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        logger.error(f"❌ Erreur MFCC sur {file_path} : {e}")
        return None

def load_dataset(data_dir):
    X, y = [], []
    for label, class_name in enumerate(['cats', 'dogs']):
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            logger.warning(f"Dossier manquant : {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                path = os.path.join(folder, fname)
                mfcc = extract_mfcc(path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(label)
    X = np.array(X)
    y = np.array(y)
    logger.info(f"📦 Dataset chargé : {len(X)} fichiers")
    return X, y


def train_audio_model(model_type, data_dir, model_path, epochs=10, use_wandb=False):
    logger.info("Loading and processing audio data...")
    X, y = load_dataset(data_dir)

    if model_type == 'transfer':
        if X.shape[2] > 173:
            X = X[:, :, :173]
        elif X.shape[2] < 173:
            pad_width = 173 - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    else:
        if X.shape[2] != 173:
            if X.shape[2] > 173:
                X = X[:, :, :173]
            else:
                pad_width = 173 - X.shape[2]
                X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X = X[..., np.newaxis]

    if len(X) == 0:
        logger.error("No data found. Please check your dataset path and files.")
        return
    
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        logger.error(f"Error during train_test_split: {e}")
        return

    logger.info(f"Input shape: {X.shape[1:]}, Model type: {model_type}")

    # Construction du modèle
    if model_type == 'simple':
        model = build_simple_audio_cnn(input_shape=X.shape[1:])
    elif model_type == 'complex':
        model = build_complex_audio_cnn(input_shape=X.shape[1:])
    elif model_type == 'transfer':
        model = build_transfer_audio_model(input_shape=X.shape[1:])
    else:
        raise ValueError("model_type must be 'simple', 'complex' or 'transfer'.")

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    # Intégration wandb
    if use_wandb:
        logger.info("🔗 Initialisation de wandb...")
        if wandb is None:
            logger.error("wandb non installé, impossible d'utiliser wandb.")
        else:
            wandb.init(project="Dogs&Cats_Project", name=f"{model_type}_image_cnn_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}", config={
                    "model_type": model_type,
                    "epochs": epochs,
                    "input_shape": X.shape[1:]
                })
            logger.info(f"✅ wandb initialisé avec le run ID : {wandb.run.id}")
            callbacks.append(WandbMetricsLogger())

    logger.info(f"🚀 Lancement de l'entraînement pour {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    final_epoch = len(history.history['loss']) - 1
    logger.info(f"📊 Final metrics (epoch {final_epoch + 1}):")
    logger.info(f"   🔹 Train Loss: {history.history['loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Train Accuracy: {history.history['accuracy'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Loss: {history.history['val_loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")

    if wandb:
        wandb.finish()

    return history
