import os
import pandas as pd

from src.image_model.train import train_model as train_image_model
from src.image_model.test import predict_on_test_images_batch

from src.audio_model.train import train_audio_model
from src.audio_model.test import predict_audio

from config.logger_config import configure_logger

logger = configure_logger()

image_model_path='models/cnn_image_model.keras'
audio_model_path='models/cnn_audio_model.keras'

def main():
    logger.info("==== DÉMARRAGE DU PIPELINE ==== 🚀")

    # 1. Entraînement du modèle image
    logger.info("➡️ Entraînement du modèle image...")
    if os.path.exists(image_model_path):
        logger.info(f"⚠️ Le modèle image existe déjà à l'emplacement {image_model_path}. Entraînement ignoré.")
    else:
        train_image_model()

    # 2. Test du modèle image
    logger.info("🔍 Prédiction sur les images de test...")
    image_results = predict_on_test_images_batch()

    # 3. Entraînement du modèle audio
    logger.info("➡️ Entraînement du modèle audio...")
    if os.path.exists(audio_model_path):
        logger.info(f"⚠️ Le modèle audio existe déjà à l'emplacement {audio_model_path}. Entraînement ignoré.")
    else:
        train_audio_model()

    # 4. Test du modèle audio
    logger.info("🔍 Prédiction sur les sons de test...")
    audio_results = predict_audio()

    # 5. Sauvegarde en CSV dans le dossier tests_results
    os.makedirs('tests_results', exist_ok=True)

    # Traitement pour les résultats image
    if hasattr(image_results, 'to_csv'):
        image_results.to_csv('tests_results/results_image.csv', index=False)
        logger.info("✅ Résultats image sauvegardés dans tests_results/results_image.csv")
    else:
        logger.warning("⚠️ image_results n'est pas un DataFrame, CSV non généré.")

    # Traitement pour les résultats audio
    try:
        if not hasattr(audio_results, 'to_csv'):
            # Essayons de convertir si c’est une liste de dict ou un dict
            if isinstance(audio_results, (list, dict)):
                audio_results = pd.DataFrame(audio_results)
            else:
                raise ValueError("audio_results ne peut pas être converti en DataFrame")
        audio_results.to_csv('tests_results/results_audio.csv', index=False)
        logger.info("✅ Résultats audio sauvegardés dans tests_results/results_audio.csv")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde des résultats audio : {e}")


    
    logger.info("✅ Pipeline terminé avec succès et résultats sauvegardés.")

if __name__ == '__main__':
    main()
