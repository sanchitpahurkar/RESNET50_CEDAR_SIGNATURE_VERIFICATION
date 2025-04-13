import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("resnet50_cedar_signature_verification.h5")

st.title("🖊️ Signature Verification App (CEDAR Dataset)")
st.write("Upload a signature image (224x224 or larger) to predict whether it is *Genuine* or *Forged*.")

uploaded_file = st.file_uploader("Choose a signature image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Signature', use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    result = "Genuine" if prediction > 0.5 else "Forged"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🧠 Prediction: **{result}**")
    st.markdown(f"### 🔍 Confidence: **{confidence*100:.2f}%**")
