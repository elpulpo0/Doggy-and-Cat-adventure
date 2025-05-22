### DOGGY AND CAT ADVENTURE

## Structure

```sh
backend/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   ├── audio/
│   │   ├── train/
│   │   └── test/
├── logs/                         # Logs de l'entraînement / inférence
├── src/
│   ├── image_model/              # Modèles et scripts liés aux images
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── model.py
│   ├── audio_model/              # Modèles et scripts liés à l'audio
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── model.py
│   ├── multimodal/               # Fusion multimodale image + audio
│   │   ├── train.py
│   │   └── utils.py
│   └── inference/                # Inference centralisée
│       ├── predictor.py
│       └── utils.py
├── app/
│   ├── api/                      # API FastAPI
│   │   ├── main.py
│   │   └── endpoints.py
│   └── tests/                    # Tests de l'application
│       ├── test_api.py
│       └── test_models.py
├── config/
│   └── logger_config.py
├── wandb/                        # Généré automatiquement par WandB
├── models/                       # Modèles entraînés (sauvegarde .keras, .h5, etc.)
├── monitoring/                   # Prometheus, Grafana, scripts de déploiement monitoring
├── .github/workflows/            # GitHub Actions (CI/CD)
├── requirements.txt
├── README.md
└── main.py                       # Point d’entrée global
frontend/                         # Frontend (streamlit, react, autre)
```

## Installation

**Clone this repository**

```bash
git clone https://github.com/elpulpo0/[...].git
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

Open 2 different terminals:

# Backend (in terminal 1)

```bash
cd backend
```

**Loggez vous sur Kaggle et WandB**

- Pour Kaggle :

Assurez vous d'avoir enregistré le token API de kaggle : https://www.kaggle.com/settings -> API
(Pour télécharger les fichiers, il faut cliquer sur "Join the competition" dans l'onglet "Data" sur https://www.kaggle.com/competitions/dogs-vs-cats/data)

- Pour WandB :

`wandb login`

**Téléchargez les fichiers**

```sh
python -m scripts.download_datas
```

**Créez, entrainez et testez les modèles**

```sh
python -m scripts.run_train_and_test
```

=======================================================

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
