import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from src.predict import predict_image

st.title("♻️ Waste Classification System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence = predict_image(image)

    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    if label in ['plastic','metal','glass','paper','cardboard']:
        st.success("♻️ Recyclable Waste")
    elif label in ['battery']:
        st.warning("⚠️ Hazardous Waste")
    else:
        st.error("🚯 Non-Recyclable Waste")