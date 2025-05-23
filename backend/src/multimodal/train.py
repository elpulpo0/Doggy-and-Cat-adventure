import os
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import librosa
from config.logger_config import configure_logger
import tensorflow_hub as hub
from tqdm import tqdm
from src.inference.utils import preprocess_image

logger = configure_logger()

def extract_yamnet_embedding(wav_path):
    waveform, sr = librosa.load(wav_path, sr=16000)
    waveform = waveform[:16000 * 10]
    waveform = waveform.flatten()
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    _, embeddings, _ = yamnet_model(waveform_tensor)
    embedding = tf.reduce_mean(embeddings, axis=0)  # moyenne sur frames
    return embedding.numpy()

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
    audio_input = Input(shape=(1024,), name="audio_input")

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
    logger.info("🔄 Début du chargement du dataset multimodal...")

    for label, class_name in enumerate(['cats', 'dogs']):
        img_folder = os.path.join(image_dir, class_name)
        audio_folder = os.path.join(audio_dir, class_name)

        logger.info(f"📂 Traitement de la classe '{class_name}' (label {label})")
        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
        audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_folder) if f.endswith('.wav')}
        common_files = img_files & audio_files

        logger.info(f"  {len(img_files)} images trouvées, {len(audio_files)} fichiers audio trouvés, {len(common_files)} fichiers communs.")

        for fname in tqdm(common_files, desc=f"Traitement {class_name}", unit="fichiers"):
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(img_folder, f"{fname}{ext}")
                if os.path.exists(p):
                    image_path = p
                    break

            if image_path is None:
                logger.warning(f"Image introuvable pour '{fname}', fichier ignoré.")
                continue

            audio_path = os.path.join(audio_folder, f"{fname}.wav")

            img = preprocess_image(image_path)
            if img is None:
                logger.warning(f"Échec du prétraitement image pour '{image_path}', fichier ignoré.")
                continue

            try:
                embedding = extract_yamnet_embedding(audio_path)
            except Exception as e:
                logger.error(f"Erreur extraction embedding audio pour '{audio_path}' : {e}")
                continue

            if embedding is None:
                logger.warning(f"Embedding audio vide pour '{audio_path}', fichier ignoré.")
                continue

            X_img.append(img)
            X_audio.append(embedding)
            y.append(label)

    X_img = np.array(X_img)
    X_audio = np.array(X_audio)
    y = np.array(y)

    logger.info(f"✅ Chargement terminé : {len(X_img)} paires image/audio valides.")
    return X_img, X_audio, y


def train_multimodal_model(
    image_model_path='models/transfer_image_model.keras',
    audio_model_path='models/yamnet_audio_model.keras',
    image_dir='data/images/train',
    audio_dir='data/audio/train',
    checkpoint_path='models/multimodal_model.keras',
    epochs=10,
    batch_size=16,
    test_size=0.2,
    random_state=42,
):
    logger.info("🔄 Chargement des données...")
    X_img, X_audio, y = load_multimodal_dataset(image_dir=image_dir, audio_dir=audio_dir)

    X_img_train, X_img_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
        X_img, X_audio, y, test_size=test_size, random_state=random_state
    )

    logger.info("📦 Construction du modèle multimodal...")
    model = build_multimodal_model(
        image_model_path=image_model_path,
        audio_model_path=audio_model_path
    )

    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')

    logger.info("🚀 Entraînement du modèle multimodal...")
    history = model.fit(
        [X_img_train, X_audio_train], y_train,
        validation_data=([X_img_val, X_audio_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint]
    )

    logger.info(f"✅ Entraînement terminé. Modèle sauvegardé : {checkpoint_path}")
    return model, history


if __name__ == "__main__":
    train_multimodal_model()
