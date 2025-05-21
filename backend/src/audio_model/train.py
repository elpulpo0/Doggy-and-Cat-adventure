import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.audio_model.model import build_audio_cnn
from config.logger_config import configure_logger
from librosa.feature import mfcc as librosa_mfcc  # import explicite avec alias

logger = configure_logger()

def extract_mfcc(file_path, max_pad_len=173):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # arguments nommés et usage direct
        if mfcc_feat.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc_feat.shape[1]
            mfcc_feat = np.pad(mfcc_feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_feat = mfcc_feat[:, :max_pad_len]
        return mfcc_feat
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def load_dataset(data_dir):
    X, y = [], []
    for label, folder in enumerate(['cat', 'dog']):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            continue
        files = os.listdir(folder_path)
        if not files:
            logger.warning(f"No files found in {folder_path}")
        for fname in files:
            if fname.endswith('.wav'):
                path = os.path.join(folder_path, fname)
                mfcc = extract_mfcc(path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(label)
                else:
                    logger.warning(f"MFCC extraction failed for {path}")
    logger.info(f"Loaded {len(X)} samples from {data_dir}")
    return np.array(X), np.array(y)

def train_audio_model(data_dir='data/audio/train', model_path='models/cnn_audio_model.keras', epochs=10):
    logger.info("Loading and processing audio data...")
    X, y = load_dataset(data_dir)

    if len(X) == 0:
        logger.error("No data found. Please check your dataset path and files.")
        return

    X = X[..., np.newaxis]  # Ajoute la dimension des canaux

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        logger.error(f"Error during train_test_split: {e}")
        return

    model = build_audio_cnn(input_shape=X.shape[1:])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    logger.info("Training CNN model on audio data...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )

    acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]
    logger.info(f"Training completed. Final accuracy: {acc:.4f}, Validation accuracy: {val_acc:.4f}")
