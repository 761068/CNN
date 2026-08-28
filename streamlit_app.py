
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("CNN.h5")
    return model

model = load_model()

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128

# IMPORTANT:
# Change these names according to your actual training folders/classes
CLASS_NAMES = ["Tumor", "No tumor"]

st.title("Brain Tumor MRI Classification")

st.write(
    "Upload an MRI scan image to classify it."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("Classifying...")

    # Resize image
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert image to RGB
    img = img.convert("RGB")

    # Convert to numpy
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    # Check model output
    if predictions.shape[-1] == 1:

        # Binary classification
        probability = float(predictions[0][0])

        if probability >= 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = probability * 100
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = (1 - probability) * 100

    else:

        # Multi-class classification
        score = tf.nn.softmax(predictions[0])

        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = float(np.max(score)) * 100

    # Show result
    st.success(
        f"This image most likely belongs to **{predicted_class}** "
        f"with a **{confidence:.2f}%** confidence."
    )

