import os
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.audio_model.train import extract_mfcc
from src.inference.utils import preprocess_image
from config.logger_config import configure_logger

logger = configure_logger()


def build_multimodal_model(image_model_path, audio_model_path):
    image_model = load_model(image_model_path)
    audio_model = load_model(audio_model_path)

    # Supprimer la dernière couche
    image_features = Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
    audio_features = Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

    # Définir les nouvelles entrées
    image_input = Input(shape=(128, 128, 3), name="image_input")
    audio_input = Input(shape=(40, 44, 1), name="audio_input")

    x = image_features(image_input)
    y = audio_features(audio_input)

    combined = concatenate([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[image_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_multimodal_dataset(image_dir='data/images/train', audio_dir='data/audio/train'):
    X_img, X_audio, y = [], [], []

    for label, class_name in enumerate(['cat', 'dog']):
        img_folder = os.path.join(image_dir, class_name)
        audio_folder = os.path.join(audio_dir, class_name)

        for fname in os.listdir(img_folder):
            if not fname.endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(img_folder, fname)
            audio_path = os.path.join(audio_folder, fname.replace('.jpg', '.wav').replace('.png', '.wav'))

            if not os.path.exists(audio_path):
                logger.warning(f"Pas de correspondance audio pour {fname}, ignoré.")
                continue

            img = preprocess_image(image_path)
            mfcc = extract_mfcc(audio_path)

            if img is None or mfcc is None:
                continue

            X_img.append(img)
            X_audio.append(mfcc)
            y.append(label)

    X_img = np.array(X_img)
    X_audio = np.array(X_audio)[..., np.newaxis]
    y = np.array(y)

    return X_img, X_audio, y


def main():
    logger.info("🔄 Chargement des données...")
    X_img, X_audio, y = load_multimodal_dataset()

    X_img_train, X_img_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        X_img, X_audio, y, test_size=0.2, random_state=42
    )

    logger.info("📦 Construction du modèle multimodal...")
    model = build_multimodal_model(
        image_model_path='models/cnn_image_model.keras',
        audio_model_path='models/cnn_audio_model.keras'
    )

    checkpoint_path = 'models/multimodal_model.keras'
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')

    logger.info("🚀 Entraînement du modèle multimodal...")
    model.fit(
        [X_img_train, X_audio_train], y_train,
        validation_data=([X_img_val, X_audio_val], y_val),
        epochs=10,
        batch_size=16,
        callbacks=[checkpoint]
    )

    logger.info("✅ Entraînement terminé. Modèle sauvegardé : models/multimodal_model.keras")


if __name__ == "__main__":
    main()
