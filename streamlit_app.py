import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('CNN.h5')
    return model

model = load_model()

# Define image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Tumor', 'No Tumor'] # Make sure these match your training classes

st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to classify if it's 'Testing' or 'Training' (representing absence or presence of tumor in this context).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert image according to model input
    if model.input_shape[-1] == 1:
        img = img.convert("L")
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        img = img.convert("RGB")
        img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(
        f"This image most likely belongs to *{predicted_class}* "
        f"with a *{confidence:.2f}%* confidence."
    )
