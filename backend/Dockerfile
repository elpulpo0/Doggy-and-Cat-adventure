# backend/Dockerfile
# ┌──────────────────────────────────────────────┐
# │  Image de base légère + build essentials     │
# └──────────────────────────────────────────────┘
FROM python:3.10-slim

# Empêche la création de .pyc et force le flush stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dossier de travail dans l’image
WORKDIR /app

# ┌──────────────────────────────────────────────┐
# │  Dépendances système utiles à CV2 / Librosa  │
# └──────────────────────────────────────────────┘
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc build-essential libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ┌──────────────────────────────────────────────┐
# │  Dépendances Python                         │
# └──────────────────────────────────────────────┘
# Copie seulement le requirements pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ┌──────────────────────────────────────────────┐
# │  Code source                                 │
# └──────────────────────────────────────────────┘
# Tout le dossier backend (app/, api/, scripts/, models/…)
COPY . .

# ┌──────────────────────────────────────────────┐
# │  Commande de démarrage                      │
# └──────────────────────────────────────────────┘
# - app.main:app  → backend/app/main.py expose l'objet `app`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]