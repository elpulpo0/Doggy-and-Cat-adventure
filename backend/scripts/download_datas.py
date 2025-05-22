import os
import shutil
import zipfile
import subprocess

from config.logger_config import configure_logger

# Configuration du logger
logger = configure_logger()

def download_and_extract_kaggle_dataset(competition_name, dest_folder):
    """Télécharge et extrait les fichiers zip du dataset Kaggle competition."""
    logger.info(f"Downloading Kaggle dataset for competition: {competition_name}")
    subprocess.run(['kaggle', 'competitions', 'download', '-c', competition_name, '-p', dest_folder], check=True)

    # Extraction du zip principal
    zip_files = [f for f in os.listdir(dest_folder) if f.endswith('.zip')]
    for zf in zip_files:
        logger.info(f"Extracting {zf}...")
        with zipfile.ZipFile(os.path.join(dest_folder, zf), 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(os.path.join(dest_folder, zf))

    # Extraction des zip secondaires (train.zip, test1.zip)
    zip_files_nested = [f for f in os.listdir(dest_folder) if f.endswith('.zip')]
    for zf in zip_files_nested:
        logger.info(f"Extracting nested zip {zf}...")
        with zipfile.ZipFile(os.path.join(dest_folder, zf), 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(os.path.join(dest_folder, zf))

    logger.info("Images dataset ready.")


def organize_images_dataset(data_dir):
    logger.info("Organizing image dataset...")
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(images_dir, 'train')
    test_dir = os.path.join(images_dir, 'test')

    os.makedirs(os.path.join(train_dir, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'dogs'), exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    extracted_train_dir = os.path.join(images_dir, 'train')
    extracted_test1_dir = os.path.join(images_dir, 'test1')

    # Déplacer uniquement les fichiers (pas les dossiers)
    if os.path.exists(extracted_train_dir):
        for f in os.listdir(extracted_train_dir):
            src = os.path.join(extracted_train_dir, f)
            if os.path.isfile(src):  # <-- important : ne déplacer que les fichiers
                if f.startswith('cat'):
                    dst = os.path.join(train_dir, 'cats', f)
                    shutil.move(src, dst)
                elif f.startswith('dog'):
                    dst = os.path.join(train_dir, 'dogs', f)
                    shutil.move(src, dst)

        # Supprimer extracted_train_dir s'il est vide après déplacement
        if not os.listdir(extracted_train_dir):
            os.rmdir(extracted_train_dir)

    # Renommer test1 en test
    if os.path.exists(extracted_test1_dir):
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.rename(extracted_test1_dir, test_dir)

    # Supprimer sampleSubmission.csv
    sample_submission_path = os.path.join(images_dir, 'sampleSubmission.csv')
    if os.path.exists(sample_submission_path):
        os.remove(sample_submission_path)

    logger.info("Images organized.")


def download_and_extract_audio_dataset(dataset_name, dest_folder):
    logger.info(f"Downloading audio dataset: {dataset_name}")
    # Si tu utilises kaggle CLI, adapte ici, par exemple :
    subprocess.run(['kaggle', 'datasets', 'download', dataset_name, '-p', dest_folder], check=True)

    zip_files = [f for f in os.listdir(dest_folder) if f.endswith('.zip')]
    for zf in zip_files:
        zip_path = os.path.join(dest_folder, zf)
        logger.info(f"Extracting {zf}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)

    logger.info("Audio dataset ready.")

def organize_audio_dataset(audio_dir):
    logger.info("Organizing audio dataset...")

    base_dir = os.path.join(audio_dir, 'cats_dogs')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # 1. Supprimer les fichiers .wav à la racine de cats_dogs (fichiers redondants)
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isfile(item_path) and item_path.endswith(".wav"):
            os.remove(item_path)
            logger.info(f"Deleted redundant file: {item_path}")

    # 2. Renommer test/test en test/dogs (s'il existe)
    if os.path.exists(os.path.join(test_dir, "test")):
        os.rename(os.path.join(test_dir, "test"), os.path.join(test_dir, "dogs"))
        logger.info("Renamed test/test to test/dogs")

    # 3. S'assurer que le dossier test/cats est bien nommé (juste au cas où)
    if os.path.exists(os.path.join(test_dir, "cats")):
        logger.info("test/cats already exists")

    # 4. Déplacer train et test dans le dossier audio/
    for subdir in ['train', 'test']:
        src = os.path.join(base_dir, subdir)
        dst = os.path.join(audio_dir, subdir)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        logger.info(f"Moved {subdir} to {audio_dir}/")

    # 5. Supprimer le dossier cats_dogs s'il est vide
    if os.path.isdir(base_dir) and not os.listdir(base_dir):
        os.rmdir(base_dir)
        logger.info("Removed empty cats_dogs directory")

    # 6. Supprimer train_test_split.csv et utils.py
    for filename in ['train_test_split.csv', 'utils.py']:
        file_path = os.path.join(audio_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")

    logger.info("Audio files organized.")

def print_dir_tree(root_dir):
    logger.info(f"\nContenu de '{root_dir}':")
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")

if __name__ == '__main__':
    backend_data_dir = 'data'

    # IMAGES
    images_data_dir = os.path.join(backend_data_dir, 'images')
    os.makedirs(images_data_dir, exist_ok=True)

    # Vérifier si le dataset est déjà extrait (ex : dossier 'train' dans images)
    if not os.path.exists(os.path.join(images_data_dir, 'train')):
        download_and_extract_kaggle_dataset('dogs-vs-cats', images_data_dir)
        organize_images_dataset(backend_data_dir)
    else:
        logger.info("Images dataset already downloaded and extracted. Skipping download.")

    # AUDIO
    audio_data_dir = os.path.join(backend_data_dir, 'audio')
    os.makedirs(audio_data_dir, exist_ok=True)

    # Vérifier si le dataset est déjà extrait (ex : dossier 'train' dans audio)
    if not os.path.exists(os.path.join(audio_data_dir, 'train')):
        download_and_extract_audio_dataset('mmoreaux/audio-cats-and-dogs', audio_data_dir)
        organize_audio_dataset(audio_data_dir)
    else:
        logger.info("Audio dataset already downloaded and extracted. Skipping download.")