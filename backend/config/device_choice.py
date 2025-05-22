import tensorflow as tf
import platform
from config.logger_config import configure_logger

logger = configure_logger()
# Détection du système d'exploitation

system = platform.system()

# Détection des GPU CUDA (NVIDIA, pour Linux/Windows)
gpus = tf.config.list_physical_devices('GPU')

# Détection des GPU Metal (Apple Silicon)
metal_gpus = tf.config.list_physical_devices('GPU') if system == 'Darwin' else []

# Gestion de la mémoire GPU (pour éviter d’allouer toute la VRAM dès le début)
def enable_memory_growth(devices):
    for gpu in devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.error(f"❌ Erreur d'activation de la croissance mémoire sur {gpu.name}: {e}")

def choose_device():
    if gpus and system != 'Darwin':
        enable_memory_growth(gpus)
        logger.info(f"✅ GPU CUDA détecté(s) et activé(s) : {[gpu.name for gpu in gpus]}")
    elif metal_gpus and system == 'Darwin':
        try:
            # Vérifie si tensorflow-metal est bien installé
            from tensorflow.python.framework import test_util
            if test_util.is_gpu_available():
                enable_memory_growth(metal_gpus)
                logger.info(f"✅ GPU Metal (Apple Silicon) détecté et activé : {[gpu.name for gpu in metal_gpus]}")
            else:
                logger.warning("⚠️ GPU Apple détecté mais non utilisable. Vérifiez que `tensorflow-metal` est installé.")
        except ImportError:
            logger.error("❌ `tensorflow-metal` non installé. Installez-le avec : `pip install tensorflow-metal`")
    else:
        logger.warning("⚠️ Aucun GPU compatible détecté, fallback sur CPU.")
