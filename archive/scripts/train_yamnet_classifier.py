import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

# Chargement du modèle YAMNet depuis TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

AUDIO_DIR = 'data/audio/train'
LABELS = ['cats', 'dogs']

X, y = [], []

# Extraction des embeddings YAMNet depuis les fichiers .wav
for idx, label in enumerate(LABELS):
    label_dir = os.path.join(AUDIO_DIR, label)
    for fname in os.listdir(label_dir):
        if fname.endswith(".wav"):
            file_path = os.path.join(label_dir, fname)
            try:
                waveform, sr = librosa.load(file_path, sr=16000)  # YAMNet attend 16kHz
                waveform = waveform[:16000 * 10]  # 10s max
                waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
                scores, embeddings, spectrogram = yamnet_model(waveform)
                mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
                X.append(mean_embedding)
                y.append(idx)
            except Exception as e:
                print(f"Erreur avec {fname}: {e}")

X = np.array(X)
y = np.array(y)

# Split
test_size = 0.2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Construction d'un petit classifieur
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Sauvegarde
model.save("models/yamnet_audio_classifier.keras")