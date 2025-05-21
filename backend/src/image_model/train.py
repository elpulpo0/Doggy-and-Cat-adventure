import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.image_model.model import build_simple_cnn
from config.logger_config import configure_logger
import os

logger = configure_logger()

def train_model(data_dir='data/images/train', model_path='models/cnn_image_model.keras', epochs=5, batch_size=32):
    logger.info(f"🧪 Début de l'entraînement du modèle CNN avec les images depuis {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"❌ Le dossier '{data_dir}' n'existe pas.")
        return

    img_size = (128, 128)

    logger.info("📂 Chargement des images d'entraînement et de validation avec image_dataset_from_directory...")

    full_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True,
        seed=123,
    )

    # Split train/validation manuellement (80%/20%)
    dataset_size = full_dataset.cardinality().numpy()
    train_size = int(0.8 * dataset_size)

    train_ds = full_dataset.take(train_size)
    val_ds = full_dataset.skip(train_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    logger.info(f"📊 Taille du dataset : total={dataset_size}, train={train_size}, val={dataset_size - train_size}")

    # Normalisation
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    logger.info("✅ Données chargées avec succès.")
    logger.info(f"🔨 Construction du modèle CNN avec input_shape=({img_size[0]}, {img_size[1]}, 3)")

    model = build_simple_cnn(input_shape=(img_size[0], img_size[1], 3))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    ]

    logger.info(f"🚀 Lancement de l'entraînement pour {epochs} epochs...")
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    logger.info("✅ Entraînement terminé.")
    logger.info(f"💾 Modèle sauvegardé dans : {model_path}")

    final_epoch = len(history.history['loss']) - 1
    logger.info(f"📊 Métriques à l'époque finale (epoch {final_epoch + 1}):")
    logger.info(f"   🔹 Train Loss: {history.history['loss'][final_epoch]:.4f} | Train Accuracy: {history.history['accuracy'][final_epoch]:.4f}")
    logger.info(f"   🔹 Val   Loss: {history.history['val_loss'][final_epoch]:.4f} | Val   Accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")

    return history
