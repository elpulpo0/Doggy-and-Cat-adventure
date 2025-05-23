from tqdm import tqdm
import os
from config.logger_config import configure_logger

logger = configure_logger("Renommage des fichiers")


def rename_files_in_directory(directory, prefix, extension):
    logger.info(f"Renommage des fichiers dans {directory} avec le préfixe '{prefix}'")
    
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()

    # Étape 1 : Renommer temporairement tous les fichiers
    tmp_names = []
    for filename in tqdm(files, desc=f"[Temp] {prefix}", unit="fichier"):
        old_path = os.path.join(directory, filename)
        tmp_name = f"__tmp__{filename}"
        tmp_path = os.path.join(directory, tmp_name)
        os.rename(old_path, tmp_path)
        tmp_names.append(tmp_name)

    # Étape 2 : Renommer proprement
    for i, tmp_name in enumerate(tqdm(tmp_names, desc=f"[Final] {prefix}", unit="fichier"), start=1):
        tmp_path = os.path.join(directory, tmp_name)
        new_name = f"{prefix}_{i}{extension}"
        new_path = os.path.join(directory, new_name)
        os.rename(tmp_path, new_path)

    logger.info(f"Tous les fichiers dans {directory} ont été renommés.")

def rename_all():
    base_dirs = [
        ('data/images/train/cats', 'cat', '.jpg'),
        ('data/images/train/dogs', 'dog', '.jpg'),
        ('data/audio/train/cat', 'cat', '.wav'),
        ('data/audio/train/dog', 'dog', '.wav'),
    ]

    for path, prefix, ext in base_dirs:
        if os.path.exists(path):
            rename_files_in_directory(path, prefix, ext)
        else:
            logger.warning(f"Dossier non trouvé : {path}")

if __name__ == "__main__":
    rename_all()
