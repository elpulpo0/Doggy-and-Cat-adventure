import os

from config.logger_config import configure_logger
from config.device_choice import choose_device

# Image training + prediction
from src.image_model.train import train_image_model
from src.image_model.test import predict_on_test_images_batch

# Multimodal YAMNet
from src.multimodal.train import train_multimodal_yamnet
from src.multimodal.test import test_multimodal_yamnet

logger = configure_logger("Pipeline Complet")
choose_device()

# === Paths ===
IMAGE_TRAIN_DIR = "data/images/train"
IMAGE_TEST_DIR = "data/images/test"
RESULTS_DIR = "tests_results"
MODEL_IMAGE_PATH = "models/transfer_cnn_image_model.keras"
MODEL_MULTIMODAL_PATH = "models/multimodal_yamnet_model.keras"

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline():
    logger.info("Pipeline complet démarré.")

    # === Étape 1 : Image transfer learning ===
    if not os.path.exists(MODEL_IMAGE_PATH):
        logger.info("Entraînement du modèle image (MobileNetV2)")
        train_image_model(model_type="transfer", data_dir=IMAGE_TRAIN_DIR, model_path=MODEL_IMAGE_PATH, use_wandb=True)
    else:
        logger.info("✅ Modèle image déjà entraîné")

    try:
        df_image = predict_on_test_images_batch(model_path=MODEL_IMAGE_PATH, test_dir=IMAGE_TEST_DIR)
        df_image['model'] = 'image_transfer'
        df_image.to_csv(f"{RESULTS_DIR}/results_image_transfer.csv", index=False)
    except Exception as e:
        logger.error(f"❌ Prédiction image échouée : {e}")

    # === Étape 2 : Multimodal (image + YAMNet audio) ===
    if not os.path.exists(MODEL_MULTIMODAL_PATH):
        logger.info("➡️ Entraînement du modèle multimodal avec YAMNet")
        train_multimodal_yamnet()
    else:
        logger.info("✅ Modèle multimodal déjà entraîné")

    try:
        test_multimodal_yamnet()
    except Exception as e:
        logger.error(f"❌ Test multimodal échoué : {e}")

    logger.info("✅ Pipeline complet terminé.")

if __name__ == "__main__":
    run_pipeline()