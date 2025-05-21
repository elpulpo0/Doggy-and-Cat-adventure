import os
import shutil

train_dir = '../data/images/train'
cats_dir = os.path.join(train_dir, 'cats')
dogs_dir = os.path.join(train_dir, 'dogs')

os.makedirs(cats_dir, exist_ok=True)
os.makedirs(dogs_dir, exist_ok=True)

print("Début du tri des images...")

for fname in os.listdir(train_dir):
    path = os.path.join(train_dir, fname)
    
    # Ignore les dossiers déjà présents
    if os.path.isdir(path):
        print(f"Ignoré (dossier) : {fname}")
        continue

    if fname.startswith('cat'):
        shutil.move(path, os.path.join(cats_dir, fname))
        print(f"Déplacé {fname} → cats/")
    elif fname.startswith('dog'):
        shutil.move(path, os.path.join(dogs_dir, fname))
        print(f"Déplacé {fname} → dogs/")
    else:
        print(f"Ignoré (nom inconnu) : {fname}")

print("Tri terminé.")
