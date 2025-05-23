import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from config.logger_config import configure_logger
from src.inference.utils import load_image_tensor

logger = configure_logger()

def extract_yamnet_embedding(wav_path):
    waveform, sr = librosa.load(wav_path, sr=16000)
    waveform = waveform[:16000 * 10]
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = hub.load("https://tfhub.dev/google/yamnet/1")(waveform_tensor)
    return tf.reduce_mean(embeddings, axis=0).numpy()

def train_multimodal_yamnet():
    img_dir = "data/images/train"
    audio_dir = "data/audio/train"
    X_img, X_audio, y = [], [], []

    logger.info("🔄 Chargement des données images + audios...")
    for label, class_name in enumerate(["cats", "dogs"]):
        img_folder = os.path.join(img_dir, class_name)
        audio_folder = os.path.join(audio_dir, class_name)

        for fname in os.listdir(img_folder):
            if not fname.endswith(".jpg"):
                continue
            base_name = os.path.splitext(fname)[0]
            audio_path = os.path.join(audio_folder, base_name + ".wav")
            if not os.path.exists(audio_path):
                # logger.warning(f"⚠️ Pas de correspondance audio pour {fname}, ignoré.")
                continue

            img_tensor = load_image_tensor(os.path.join(img_folder, fname))
            audio_embedding = extract_yamnet_embedding(audio_path)

            X_img.append(img_tensor)
            X_audio.append(audio_embedding)
            y.append(label)

    X_img = np.array(X_img)
    X_audio = np.array(X_audio)
    y = np.array(y)

    X_img_train, X_img_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        X_img, X_audio, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("📦 Chargement du modèle image pré-entraîné (MobileNetV2)")
    image_model = load_model("models/transfer_cnn_image_model.keras")
    image_model.trainable = False  # on freeze

    image_input = Input(shape=(128, 128, 3))
    image_features = image_model(image_input)

    audio_input = Input(shape=(1024,))
    audio_features = Dense(128, activation='relu')(audio_input)

    merged = Concatenate()([image_features, audio_features])
    merged = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[image_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("🏋️ Entraînement du modèle multimodal en cours...")
    model.fit(
        [X_img_train, X_audio_train],
        y_train,
        validation_data=([X_img_val, X_audio_val], y_val),
        epochs=10,
        batch_size=32
    )

    logger.info("💾 Sauvegarde finale dans models/multimodal_yamnet_model.keras")
    model.save("models/multimodal_yamnet_model.keras")