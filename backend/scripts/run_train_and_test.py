import os
import pandas as pd

from src.image_model.train import train_image_model
from src.image_model.test import predict_on_test_images_batch

from src.audio_model.train import train_audio_model
from src.audio_model.test import predict_audio

from config.logger_config import configure_logger
from config.device_choice import choose_device

logger = configure_logger("Script d'entrainement et de tests")


simple_image_model_path='models/simple_cnn_image_model.keras'
complex_image_model_path='models/simple_cnn_image_model.keras'
transfer_model_path = 'models/transfer_cnn_image_model.keras'
audio_model_path='models/cnn_audio_model.keras'


def main():
    logger.info("==== DÉMARRAGE DU PIPELINE ==== 🚀")

    # Défini le device pour l'entrainement
    choose_device()
    
    test_dir = 'data/images/test'

    results_dir = 'tests_results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Entraînement des modèles image
    model_infos = [
        ("simple", simple_image_model_path),
        ("complex", complex_image_model_path),
        ("transfer", transfer_model_path),
    ]
    
    for model_type, model_path in model_infos:
        if not os.path.exists(model_path):
            logger.info(f"➡️ Entraînement du modèle {model_type}...")
            train_image_model(model_type=model_type, data_dir="data/images/train", model_path=model_path, use_wandb=True)
        else:
            logger.info(f"⚠️ Le modèle {model_type} existe déjà, entraînement ignoré.")

    # 2. Prédiction sur les images de test
    all_results = []
    for model_type, model_path in model_infos:
        logger.info(f"🔍 Prédiction avec le modèle {model_type}...")
        df = predict_on_test_images_batch(model_path=model_path, test_dir=test_dir)
        df['model'] = model_type
        df.to_csv(f"{results_dir}/results_image_{model_type}.csv", index=False)
        all_results.append(df)

    # Fusion des résultats image
    image_results = pd.concat(all_results, ignore_index=True)

    # 3. Entraînement du modèle audio
    logger.info("➡️ Entraînement du modèle audio...")
    if os.path.exists(audio_model_path):
        logger.info(f"⚠️ Le modèle audio existe déjà. Entraînement ignoré.")
    else:
        train_audio_model()

    # 4. Test du modèle audio
    logger.info("🔍 Prédiction sur les sons de test...")
    audio_results = predict_audio()

    # 5. Sauvegarde des résultats dans le dossier tests_results

    # Sauvegarde résultats image
    if isinstance(image_results, pd.DataFrame):
        image_results.to_csv(f'{results_dir}/results_image_concat.csv', index=False)
        logger.info("✅ Résultats image sauvegardés.")
    else:
        logger.warning("⚠️ image_results n'est pas un DataFrame, CSV non généré.")

    # Sauvegarde résultats audio
    try:
        if not isinstance(audio_results, pd.DataFrame):
            if isinstance(audio_results, (list, dict)):
                audio_results = pd.DataFrame(audio_results)
            else:
                raise ValueError("audio_results ne peut pas être converti en DataFrame")
        audio_results.to_csv(f'{results_dir}/results_audio.csv', index=False)
        logger.info("✅ Résultats audio sauvegardés.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde des résultats audio : {e}")

    logger.info("✅ Pipeline terminé avec succès et résultats sauvegardés.")

if __name__ == '__main__':
    main()
