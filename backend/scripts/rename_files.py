from tqdm import tqdm
import os
from config.logger_config import configure_logger

logger = configure_logger("Renommage des fichiers")

def ensure_plural_folder_names(base_dir):
    """S'assure que les dossiers sont au pluriel ('cat' -> 'cats', 'dog' -> 'dogs')."""
    corrections = {'cat': 'cats', 'dog': 'dogs'}
    for singular, plural in corrections.items():
        old_path = os.path.join(base_dir, singular)
        new_path = os.path.join(base_dir, plural)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            logger.info(f"Dossier renommé : {old_path} -> {new_path}")

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
    # Correction des noms de dossiers s'ils sont au singulier
    ensure_plural_folder_names('data/images/train')
    ensure_plural_folder_names('data/audio/train')

    base_dirs = [
        ('data/images/train/cats', 'cat', '.jpg'),
        ('data/images/train/dogs', 'dog', '.jpg'),
        ('data/audio/train/cats', 'cat', '.wav'),
        ('data/audio/train/dogs', 'dog', '.wav'),
    ]

    for path, prefix, ext in base_dirs:
        if os.path.exists(path):
            rename_files_in_directory(path, prefix, ext)
        else:
            logger.warning(f"Dossier non trouvé : {path}")

if __name__ == "__main__":
    rename_all()
