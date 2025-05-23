from keras.preprocessing import image
from config.logger_config import configure_logger

logger = configure_logger()

def load_image_tensor(image_path, target_size=(128, 128)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Erreur chargement image : {e}")
        return None