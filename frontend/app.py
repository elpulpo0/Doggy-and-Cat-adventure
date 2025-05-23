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

# Tabs pour image vs audio
mode = st.tabs(["🖼️ Image", "🎧 Audio"])

with mode[0]:
    st.markdown("""
    Téléverse une image et découvre si notre modèle pense que c’est un **🐶 chien** ou un **🐱 chat**.
    """)

    uploaded_file = st.file_uploader("📤 Upload une image", type=["jpg", "jpeg", "png"], key="image")

    if uploaded_file:
        st.image(uploaded_file, caption="🖼️ Image sélectionnée", use_container_width=True)

        st.markdown("---")
        st.markdown("### Résultat de la prédiction")

        if st.button("Lancer la prédiction", key="predict_img"):
            with st.container():
                st_lottie(load_lottiefile("https://assets7.lottiefiles.com/private_files/lf30_obidsi0t.json"), height=200)
                st.markdown("### 🤖 L’IA est en train d’analyser ton image...")

                sleep(2)

                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                try:
                    response = requests.post("http://localhost:8000/predict/image", files=files)
                    response.raise_for_status()
                    result = response.json()

                    label = result['prediction'].upper()
                    prediction_raw = float(result['confidence'])
                    confidence = round(prediction_raw * 100, 2) if label.lower() == "chien" else round((1 - prediction_raw) * 100, 2)

                    st.success(f"✅ **{label}** détecté avec une confiance de {confidence} %")
                    st.progress(min(int(confidence), 100))

                    st.markdown("🔬 *Ce modèle n’est pas infaillible. Il donne une estimation basée sur l’apprentissage supervisé.*")

                except requests.exceptions.RequestException as e:
                    st.error("❌ Erreur lors de l'appel à l'API.")
                    st.exception(e)

with mode[1]:
    st.markdown("""
    Téléverse un fichier audio `.wav` contenant un miaulement ou un aboiement, et laisse l’IA deviner.
    """)

    uploaded_audio = st.file_uploader("📤 Upload un son", type=["wav"], key="audio")

    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")

        if st.button("Lancer la prédiction", key="predict_audio"):
            with st.spinner("🎧 Analyse du fichier audio en cours..."):
                try:
                    files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)}
                    response = requests.post("http://localhost:8000/predict/audio-yamnet", files=files)
                    response.raise_for_status()
                    result = response.json()

                    label = result['prediction'].upper()
                    prediction_raw = float(result['confidence'])
                    confidence = round(prediction_raw * 100, 2) if label.lower() == "chien" else round((1 - prediction_raw) * 100, 2)

                    st.success(f"🔊 **{label}** détecté avec une confiance de {confidence} %")
                    st.progress(min(int(confidence), 100))

                except requests.exceptions.RequestException as e:
                    st.error("❌ Erreur lors de l'appel à l'API audio.")
                    st.exception(e)