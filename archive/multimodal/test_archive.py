import os
from src.inference.predictor import MultimodalPredictor
from config.logger_config import configure_logger

logger = configure_logger()

def batch_test():
    predictor = MultimodalPredictor(model_path='models/multimodal_model.keras')

    chiens = [1, 2, 3, 4, 12, 17, 18]
    chats = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20]

    image_test_dir = 'data/images/test'
    audio_test_dir = 'data/audio/test'

    dog_audio_files = sorted(os.listdir(os.path.join(audio_test_dir, 'dogs')))
    cat_audio_files = sorted(os.listdir(os.path.join(audio_test_dir, 'cats')))

    assert len(dog_audio_files) >= len(chiens), f"Pas assez de fichiers audio chiens"
    assert len(cat_audio_files) >= len(chats), f"Pas assez de fichiers audio chats"

    logger.info("=== Début des tests Chiens ===")
    for img_num, audio_file in zip(chiens, dog_audio_files):
        image_path = f"{image_test_dir}/{img_num}.jpg"
        audio_path = os.path.join(audio_test_dir, 'dogs', audio_file)
        try:
            class_id, conf, label = predictor.predict(image_path, audio_path)
            logger.info(f"Paire {img_num}.jpg / {audio_file} --> Classe={class_id} ({label}), Confiance={conf*100:.2f}%")
        except Exception as e:
            logger.error(f"Erreur sur test chiens {img_num}: {e}")

    logger.info("=== Début des tests Chats ===")
    for img_num, audio_file in zip(chats, cat_audio_files):
        image_path = f"{image_test_dir}/{img_num}.jpg"
        audio_path = os.path.join(audio_test_dir, 'cats', audio_file)
        try:
            class_id, conf, label = predictor.predict(image_path, audio_path)
            logger.info(f"Paire {img_num}.jpg / {audio_file} --> Classe={class_id} ({label}), Confiance={conf*100:.2f}%")
        except Exception as e:
            logger.error(f"Erreur sur test chats {img_num}: {e}")

if __name__ == "__main__":
    batch_test()