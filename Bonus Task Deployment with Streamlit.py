# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as

model = tf.keras.models.load_model("mnist_model.h5")

st.title("MNIST Digit Classifier")
uploaded = st.file_uploader("Upload an image...", type=["png", "jpg"])

if uploaded:
    image = Image.open(uploaded).convert("L").resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)
    pred = model.predict(img_array).argmax()
    st.image(image, caption=f"Prediction: {pred}", width=150)
