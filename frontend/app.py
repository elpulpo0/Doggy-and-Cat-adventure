import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json
from time import sleep

# Configuration de la page
st.set_page_config(page_title="Chien ou Chat 🐶🐱", layout="centered", page_icon="🐾")


# Bannière animée en Lottie (si streamlit-lottie installé)
def load_lottiefile(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None


lottie_anim = load_lottiefile("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json") # paw animation

st_lottie(lottie_anim, height=200, key="header_anim")

st.title("🐾 Prédiction IA : Chien ou Chat")

st.markdown("""
Bienvenue sur notre mini-app d’intelligence artificielle.
Téléverse une image et découvre si notre modèle pense que c’est un **🐶 chien** ou un **🐱 chat**.

*Modèle basé sur MobileNetV2, entraîné sur des milliers d’images.*
""")

uploaded_file = st.file_uploader("📤 Upload une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Image sélectionnée", use_container_width=True)

    st.markdown("---")
    st.markdown("### Résultat de la prédiction")

    if st.button("Lancer la prédiction"):
        with st.container():
            st_lottie(load_lottiefile("https://assets7.lottiefiles.com/private_files/lf30_obidsi0t.json"), height=200)
            st.markdown("### 🤖 L’IA est en train d’analyser ton image...")
            
            # ✅ Pause volontaire
            sleep(2)  # ← délai visible de l'animation avant que le résultat apparaisse

            # ✅ D'abord, préparer le fichier
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            try:
                # ✅ Ensuite, appel API
                response = requests.post("http://localhost:8000/predict/image", files=files)
                response.raise_for_status()
                result = response.json()

                label = result['prediction'].upper()
                prediction_raw = float(result['confidence'])

                # si le modèle prédit CHAT, on prend 1 - score
                if label.lower() == "chien":
                    confidence = round(prediction_raw * 100, 2)
                else:
                    confidence = round((1 - prediction_raw) * 100, 2)

                st.success(f"✅ **{label}** détecté avec une confiance de {confidence} %")
                st.progress(min(int(confidence), 100))

                st.markdown("*Ce modèle n’est pas infaillible. Il donne une estimation basée sur l’apprentissage supervisé.*")

            except requests.exceptions.RequestException as e:
                st.error("❌ Erreur lors de l'appel à l'API.")
                st.exception(e)

else:
    st.info("Pour commencer, téléverse une image.")