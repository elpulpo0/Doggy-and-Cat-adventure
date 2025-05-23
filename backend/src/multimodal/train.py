import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.audio_model.train import extract_mfcc
from src.inference.utils import preprocess_image
from config.logger_config import configure_logger

logger = configure_logger()

def clone_and_rename(model, prefix):
    def clone_function(layer):
        config = layer.get_config()
        config['name'] = prefix + config['name']
        return layer.__class__.from_config(config)
    new_model = tf.keras.models.clone_model(model, clone_function=clone_function)
    new_model.set_weights(model.get_weights())
    return new_model

def build_multimodal_model(image_model_path, audio_model_path):
    image_model = load_model(image_model_path)
    audio_model = load_model(audio_model_path)

    # Cloner et renommer les couches pour éviter les conflits de noms
    image_model = clone_and_rename(image_model, "img_")
    audio_model = clone_and_rename(audio_model, "aud_")

    # Définir les nouvelles entrées
    image_input = Input(shape=(128, 128, 3), name="image_input")
    audio_input = Input(shape=(40, 173, 3), name="audio_input")

    # Appliquer manuellement chaque couche (sauf la dernière)
    x = image_input
    for layer in image_model.layers[:-1]:
        x = layer(x)

    y = audio_input
    for layer in audio_model.layers[:-1]:
        y = layer(y)

    # Fusionner les sorties
    combined = concatenate([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[image_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_multimodal_dataset(image_dir='data/images/train', audio_dir='data/audio/train'):
    X_img, X_audio, y = [], [], []

    for label, class_name in enumerate(['cats', 'dogs']):
        img_folder = os.path.join(image_dir, class_name)
        audio_folder = os.path.join(audio_dir, class_name)

        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
        audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_folder) if f.endswith('.wav')}

        common_files = img_files & audio_files

        if not common_files:
            logger.warning(f"Aucune paire image/audio trouvée pour la classe {class_name}")

        for fname in common_files:
            image_path = os.path.join(img_folder, f"{fname}.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(img_folder, f"{fname}.png")
            if not os.path.exists(image_path):
                image_path = os.path.join(img_folder, f"{fname}.jpeg")
            if not os.path.exists(image_path):
                logger.warning(f"Image introuvable : {fname}, ignoré.")
                continue

            audio_path = os.path.join(audio_folder, f"{fname}.wav")

            img = preprocess_image(image_path)
            mfcc = extract_mfcc(audio_path)

            if img is None or mfcc is None:
                continue

            X_img.append(img)
            X_audio.append(mfcc)
            y.append(label)

    X_img = np.array(X_img)
    X_audio = np.array(X_audio)

    # Ajuster la largeur à 173 frames (padding ou découpage)
    if X_audio.shape[2] > 173:
        X_audio = X_audio[:, :, :173]
    elif X_audio.shape[2] < 173:
        pad_width = 173 - X_audio.shape[2]
        X_audio = np.pad(X_audio, ((0,0), (0,0), (0,pad_width)), mode='constant')

    # Ajouter 3 canaux pour correspondre à l'entrée du modèle audio
    X_audio = np.repeat(X_audio[..., np.newaxis], 3, axis=-1)
    
    y = np.array(y)

    logger.info(f"🔍 {len(X_img)} paires image/audio valides chargées.")
    return X_img, X_audio, y

def main():
    logger.info("🔄 Chargement des données...")
    X_img, X_audio, y = load_multimodal_dataset()

    X_img_train, X_img_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        X_img, X_audio, y, test_size=0.2, random_state=42
    )

    logger.info("📦 Construction du modèle multimodal...")
    model = build_multimodal_model(
        image_model_path='models/transfer_cnn_image_model.keras',
        audio_model_path='models/transfer_cnn_audio_model.keras'
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
