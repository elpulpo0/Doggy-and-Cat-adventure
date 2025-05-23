### DOGGY AND CAT ADVENTURE

## Structure

```sh
backend/
├── archive/                       # Legcy code
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

## Lancer le projet – Backend (Terminal 1)

```bash
cd backend
```

### Connexion aux services externes

* **Kaggle** : place ton fichier `kaggle.json` dans `~/.kaggle/`
  (obtenu sur [https://www.kaggle.com/account](https://www.kaggle.com/account) → "Create API Token")
* **WandB** (si utilisé) :

```bash
wandb login
```

---

### Télécharger les données

```bash
python -m scripts.download_datas
```

### Entraîner les modèles image/audio

```bash
python -m scripts.run_train_and_test
```

---

### Lancer l’API FastAPI

```bash
uvicorn main:app --reload
```

Cela démarre une API REST sur :

* Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
* Endpoint actif : `POST /predict/image`
  → prend un fichier `.jpg/.png` et retourne `{"prediction": ..., "confidence": ...}`

---

## Frontend Streamlit (Terminal 2)

```bash
cd frontend
streamlit run app.py
```

### Ce que fait l'interface :

* Permet d’**uploader une image**
* Affiche l’image avec une **animation stylée (Lottie)**
* Envoie la requête à l’API FastAPI `/predict/image`
* Affiche la **prédiction** (`Chien` ou `Chat`) + **niveau de confiance (%)**
* Barre de progression + messages UX pour guider l’utilisateur

---

### Interface avec 3 onglets : Image 🖼️, Audio 🎵, Multimodal 🧹

* Upload fichiers
* Appel à l’API appropriée
* Affichage prédiction + confiance + UX animée

---

### À noter :

* Le frontend Streamlit ne contient pas de logique métier : il **consomme l’API backend**.
* Le modèle est chargé côté FastAPI pour des raisons de sécurité et de scalabilité.
* L’inférence est centralisée pour un futur support audio/multimodal.

---

## Modèles IA

| Type       | Modèle                      | Détail                                 |
| ---------- | --------------------------- | -------------------------------------- |
| Image      | MobileNetV2                 | Fine-tuné sur images chiens/chats      |
| Audio      | YAMNet                      | Classifie aboiement vs miaulement      |
| Multimodal | Fusion MobileNetV2 + YAMNet | Fusion Dense sur embeddings concaténés |

---

## Avancement

* ✅ Modèles image/audio/multimodal prêts
* ✅ API FastAPI avec endpoints actifs
* ✅ Streamlit avec 3 modes fonctionnels
* ✅ Bonne précision (multimodal > 95 %)

---

## Prochaines étapes

* 🔹 Ajout d’un benchmark comparatif
* 🔹 Nettoyage des fichiers orphelins
* 🔹 Prédiction "aucun des deux"
* 🔹 Déploiement cloud (Render / HF Spaces)

---

## Stack technique

| Composant | Tech                        |
| --------- | --------------------------- |
| Backend   | FastAPI, TensorFlow, YAMNet |
| Frontend  | Streamlit, Lottie           |
| Modèles   | CNN, MobileNetV2, YAMNet    |
| CI/CD     | GitHub Actions              |

---

## Crédits

Projet réalisé dans le cadre de la formation de \[Développeur en Intelligence Artificielle] à \[Simplon.co]
- Mathieu Soussignan
- Chris El-Pulpo
- Arnaud Boy
- Anthony Vallad