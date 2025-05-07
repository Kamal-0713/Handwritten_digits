import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit (0–9).")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image).resize((28, 28))
    st.image(image, caption="Processed Image", width=150)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    st.success(f"Predicted Digit: {np.argmax(prediction)}")