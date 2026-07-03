import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

@st.cache_resource
def charger_modele():
    if not os.path.exists("model_cats_dogs.keras"):
        url = "https://drive.google.com/uc?id=1qLg0Kq81mO4WXtqjd4oKGh4hkEQxOWsy"
        gdown.download(url, "model_cats_dogs.keras", quiet=False)
    return load_model("model_cats_dogs.keras")

model = charger_modele()

st.title("Classification Chien vs Chat")
st.write("Chargez une image, le modele predit s'il s'agit d'un chien ou d'un chat.")

fichier = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if fichier is not None:
    st.image(fichier, caption="Image chargee", use_container_width=True)
    img = load_img(fichier, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    proba = model.predict(img_array)[0][0]
    if proba > 0.5:
        st.subheader(f"C'est un CHIEN ({proba*100:.1f}% de confiance)")
    else:
        st.subheader(f"C'est un CHAT ({(1-proba)*100:.1f}% de confiance)")