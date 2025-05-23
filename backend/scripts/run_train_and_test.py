import os
import pandas as pd

from src.image_model.train import train_image_model
from src.image_model.test import predict_on_test_images_batch

from src.audio_model.train import train_audio_model, train_yamnet_model
from src.audio_model.test import predict_audio

from src.multimodal.train import train_multimodal_model

from config.logger_config import configure_logger
from config.device_choice import choose_device

logger = configure_logger("Script d'entrainement et de tests")

# Modèles image
simple_image_model_path = 'models/simple_cnn_image_model.keras'
complex_image_model_path = 'models/complex_cnn_image_model.keras'
transfer_image_model_path = 'models/transfer_image_model.keras'


# Modèles audio
simple_audio_model_path = 'models/simple_cnn_audio_model.keras'
complex_audio_model_path = 'models/complex_cnn_audio_model.keras'
transfer_audio_model_path = 'models/transfer_audio_model.keras'
yamnet_audio_model_path = 'models/yamnet_audio_model.keras'


def main():
    logger.info("==== DÉMARRAGE DU PIPELINE ==== 🚀")

    choose_device()
    
    test_image_dir = 'data/images/test'
    test_audio_dir = 'data/audio/test'
    image_train_dir = "data/images/train"
    audio_train_dir = "data/audio/train"
    results_dir = 'tests_results'
    os.makedirs(results_dir, exist_ok=True)

    # === 1. Entraînement des modèles image ===
    image_model_infos = [
        # ("simple", simple_image_model_path),
        # ("complex", complex_image_model_path),
        ("transfer", transfer_image_model_path),
    ]
    
    for model_type, model_path in image_model_infos:
        if not os.path.exists(model_path):
            logger.info(f"➡️ Entraînement du modèle image {model_type}...")
            train_image_model(model_type=model_type, data_dir=image_train_dir, model_path=model_path, use_wandb=True)
        else:
            logger.info(f"⚠️ Le modèle image {model_type} existe déjà, entraînement ignoré.")

    # === 2. Prédictions sur les images test ===
    all_image_results = []
    for model_type, model_path in image_model_infos:
        logger.info(f"🔍 Prédiction avec le modèle image {model_type}...")

        try:
            df = predict_on_test_images_batch(model_path=model_path, test_dir=test_image_dir)
            if df is not None and not df.empty:
                df['model'] = f"image_{model_type}"
                df.to_csv(f"{results_dir}/results_image_{model_type}.csv", index=False)
                all_image_results.append(df)
            else:
                logger.warning(f"⚠️ Résultat vide pour le modèle image {model_type}.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction avec le modèle image {model_type} : {e}")

    # === 3. Entraînement des modèles audio ===
    audio_model_infos = [
        # ("simple", simple_audio_model_path),
        # ("complex", complex_audio_model_path),
        # ("transfer", transfer_audio_model_path),
    ]

    for model_type, model_path in audio_model_infos:
        if not os.path.exists(model_path):
            logger.info(f"➡️ Entraînement du modèle audio {model_type}...")
            train_audio_model(model_type=model_type, data_dir=audio_train_dir, model_path=model_path, use_wandb=True)
        else:
            logger.info(f"⚠️ Le modèle audio {model_type} existe déjà. Entraînement ignoré.")

    # === 3bis. Entraînement du modèle audio yamnet ===
    if not os.path.exists(yamnet_audio_model_path):
        logger.info(f"➡️ Entraînement du modèle audio yamnet...")
        train_yamnet_model(data_dir=audio_train_dir, model_path=yamnet_audio_model_path, use_wandb=True)
    else:
        logger.info(f"⚠️ Le modèle audio yamnet existe déjà. Entraînement ignoré.")

    # === 4. Prédictions audio ===
    all_audio_results = []
    for model_type, model_path in audio_model_infos:
        logger.info(f"🔍 Prédiction avec le modèle audio {model_type}...")

        try:
            results = predict_audio(model_path=model_path, model_type=model_type, test_dir=test_audio_dir, batch_size=20)
            if results:
                df = pd.DataFrame(results)
                df['model'] = f"audio_{model_type}"
                df.to_csv(f"{results_dir}/results_audio_{model_type}.csv", index=False)
                all_audio_results.append(df)
            else:
                logger.warning(f"⚠️ Résultat vide pour le modèle audio {model_type}.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction avec le modèle audio {model_type} : {e}")
    
    # === 4bis. Prédiction avec yamnet ===
    if os.path.exists(yamnet_audio_model_path):
        logger.info("🔍 Prédiction avec le modèle audio yamnet...")

        try:
            results = predict_audio(model_path=yamnet_audio_model_path, model_type="yamnet", test_dir=test_audio_dir, batch_size=20)
            if results:
                df = pd.DataFrame(results)
                df['model'] = "audio_yamnet"
                df.to_csv(f"{results_dir}/results_audio_yamnet.csv", index=False)
                all_audio_results.append(df)
            else:
                logger.warning("⚠️ Résultat vide pour le modèle audio yamnet.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction avec le modèle audio yamnet : {e}")
    else:
        logger.warning("⚠️ Modèle audio yamnet non trouvé pour la prédiction.")

    # === 5. Fusion et sauvegardes des résultats ===
    # for result_type, all_results in [("image", all_image_results), ("audio", all_audio_results)]:
    #     try:
    #         result_df = pd.concat(all_results, ignore_index=True)
    #         result_df.to_csv(f'{results_dir}/results_{result_type}_concat.csv', index=False)
    #         logger.info(f"✅ Résultats {result_type} sauvegardés.")
    #     except Exception as e:
    #         logger.error(f"❌ Erreur lors de la sauvegarde des résultats {result_type} : {e}")


    # === 6. Entraînement du modèle multimodal ===
    
    if not os.path.exists('models/multimodal_model.keras'):
        logger.info("➡️ Entraînement du modèle multimodal...")
        train_multimodal_model(
            image_model_path=transfer_image_model_path,
            audio_model_path=yamnet_audio_model_path,
            image_dir=image_train_dir,
            audio_dir=audio_train_dir,
            model_save_path='models/multimodal_model.keras',
            epochs=10,
            batch_size=16
        )
    else:
        logger.info("⚠️ Le modèle multimodal existe déjà, entraînement ignoré.")

        logger.info("✅ Pipeline terminé avec succès et résultats sauvegardés.")

if __name__ == '__main__':
    main()
