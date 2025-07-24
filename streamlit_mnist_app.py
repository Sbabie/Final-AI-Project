# streamlit_mnist_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Title
st.title("MNIST Handwritten Digit Classifier")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")  # Ensure this file is present in the same folder
    return model

model = load_model()

# Upload image
uploaded = st.file_uploader("MNIST digit_sample.png )", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)

    # Prepare image
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img_array)
    predicted_class = pred.argmax()

    st.subheader(f"Prediction: {predicted_class}")
