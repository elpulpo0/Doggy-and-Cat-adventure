### DOGGY AND CAT ADVENTURE

## Structure

```sh
backend/
├── data/
│   ├── images/          # Dataset image (chats/chiens)
├── ├── ├── train/
├── ├── └── test/
│   ├── audio/           # Dataset audio (aboiement, miaulement, pas)
│   ├── processed/       # Données préparées (spectrogrammes, splits, etc.)
│   └── logs/
├── src/
│   ├── image_model/     # Modèles et entraînement vision (CNN, MobileNet...)
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── audio_model/     # Modèles et entraînement audio (MFCC, CNN audio...)
│   │   ├── train.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── multimodal/      # Fusion image + audio
│   │   ├── train.py
│   │   ├── model.py
│   │   └── utils.py
│   └── inference/       # Prétraitement + chargement modèles pour l'inférence
│       ├── predictor.py
│       └── audio_utils.py
├── app/
│   ├── api/             # API FastAPI
│   │   ├── main.py
│   │   └── endpoints.py
│   └── tests/           # Tests unitaires/fonctionnels
│       ├── test_api.py
│       └── test_models.py
├── config/
│   ├── config.yaml      # Paramètres modèles / chemin data
│   └── logging.yaml     # Configuration des logs
├── mlruns/              # Dossier généré par MLflow (automatique)
├── models/              # Modèles entraînés sauvegardés
├── monitoring/          # Config Prometheus, Grafana (déploiement)
├── .github/workflows/   # CI/CD GitHub Actions
├── requirements.txt
├── README.md
└── main.py              # Point d’entrée global pour exécution / test
frontend/
```

## Installation

**Clone this repository**

```bash
git clone https://github.com/elpulpo0/[...].git
```

**Téléchargez les fichiers**

Assurez vous d'abord d'avoir enregistré le token API de kaggle : https://www.kaggle.com/settings -> API

```sh
cd backend && python scripts/download_datas.py
```

Open 2 different terminals:

# Backend (in terminal 1)

```bash
cd backend
```

**Create a virtual environnement**

```bash
python -m venv .venv
```

**Connect to the virtual environnement**

```bash
source .venv/Scripts/activate
```

**Upgrade pip and install librairies**

```bash
python.exe -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

**Run the app**

```bash
uvicorn run:app --reload
```

# Frontend (in terminal 2)

```bash
cd frontend
```

**Install dependencies**

```bash
npm install
```

**Run the app**

```bash
npm run dev
```
