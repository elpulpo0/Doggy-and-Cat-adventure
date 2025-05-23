import os
import datetime
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.audio_model.models import (
    build_simple_audio_cnn,
    build_complex_audio_cnn,
    build_transfer_audio_model
)
from config.logger_config import configure_logger

try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
except ImportError:
    wandb = None

logger = configure_logger()

def extract_mfcc(file_path, max_pad_len=173, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_pad_len - mfcc.shape[1]))), mode='constant')[:, :max_pad_len]
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
    X, y = np.array(X), np.array(y)
    logger.info(f"📦 Dataset chargé : {len(X)} fichiers")
    return X, y

def train_audio_model(model_type, data_dir, model_path, epochs=10, use_wandb=False):
    logger.info("🔍 Chargement et traitement des données audio...")
    X, y = load_dataset(data_dir)

    if len(X) == 0:
        logger.error("❌ Aucun fichier audio trouvé.")
        return

    if model_type == 'transfer':
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    else:
        X = X[..., np.newaxis]

    if X.shape[2] != 173:
        pad_width = 173 - X.shape[2]
        X = np.pad(X, ((0, 0), (0, 0), (0, max(0, pad_width))), mode='constant')[:, :, :173]

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as e:
        logger.error(f"❌ Erreur lors du split des données : {e}")
        return

    logger.info(f"📐 Input shape : {X.shape[1:]}, Modèle : {model_type}")

    if model_type == 'simple':
        model = build_simple_audio_cnn(input_shape=X.shape[1:])
    elif model_type == 'complex':
        model = build_complex_audio_cnn(input_shape=X.shape[1:])
    elif model_type == 'transfer':
        model = build_transfer_audio_model(input_shape=X.shape[1:])
    else:
        raise ValueError("❌ model_type doit être 'simple', 'complex' ou 'transfer'.")

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    if use_wandb and wandb is not None:
        wandb.init(
            project="Dogs&Cats_Project",
            name=f"{model_type}_audio_cnn_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
            config={"model_type": model_type, "epochs": epochs, "input_shape": X.shape[1:]}
        )
        logger.info(f"✅ wandb initialisé : {wandb.run.id}")
        callbacks.append(WandbMetricsLogger())

    logger.info(f"🚀 Début de l'entraînement ({epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    final_epoch = len(history.history['loss']) - 1
    logger.info(f"📊 Résultats finaux (epoch {final_epoch + 1}):")
    logger.info(f"   🔹 Train Loss: {history.history['loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Train Accuracy: {history.history['accuracy'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Loss: {history.history['val_loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")

    if wandb:
        wandb.finish()

    return history

def train_yamnet_model(data_dir, model_path, max_duration=10, use_wandb=False):
    logger.info("🔊 Entraînement du modèle YAMNet...")
    LABELS = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    X, y = [], []

    for idx, label in enumerate(LABELS):
        label_dir = os.path.join(data_dir, label)
        for fname in os.listdir(label_dir):
            if fname.endswith(".wav"):
                file_path = os.path.join(label_dir, fname)
                try:
                    waveform, sr = librosa.load(file_path, sr=16000)
                    waveform = waveform[:16000 * max_duration]
                    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
                    X.append(mean_embedding)
                    y.append(idx)
                except Exception as e:
                    logger.error(f"❌ Erreur avec {fname}: {e}")

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        logger.error("❌ Aucune donnée extraite.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []

    if use_wandb and wandb is not None:
        wandb.init(
            project="Dogs&Cats_Project",
            name=f"yamnet_audio_classifier_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
            config={"model_type": "yamnet", "input_shape": (1024,), "epochs": 10}
        )
        logger.info(f"✅ wandb initialisé pour YAMNet : {wandb.run.id}")
        callbacks.append(WandbMetricsLogger())

    logger.info("🚀 Entraînement du classifieur YAMNet...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    final_epoch = len(history.history['loss']) - 1
    logger.info(f"📊 Résultats finaux (epoch {final_epoch + 1}):")
    logger.info(f"   🔹 Train Loss: {history.history['loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Train Accuracy: {history.history['accuracy'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Loss: {history.history['val_loss'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"✅ Modèle YAMNet sauvegardé à : {model_path}")

    if wandb:
        wandb.finish()

    return history
