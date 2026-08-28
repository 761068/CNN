import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# Define image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Testing', 'Training'] # Make sure these match your training classes

st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to classify if it's 'Testing' or 'Training' (representing absence or presence of tumor in this context).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((IMG_HEIGHT, IMG_WIDTH)))
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    # If the image is grayscale, convert to 3 channels
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    elif img_array.shape[-1] == 4: # Handle RGBA images
        img_array = img_array[:,:,:,:3] # Drop the alpha channel

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"This image most likely belongs to **{predicted_class}** with a **{confidence:.2f}%** confidence.")

# Instructions for GitHub deployment
st.markdown("""
--- 
### Deployment Instructions (GitHub)

1.  **Create a new GitHub repository** for your Streamlit app.
2.  **Add all files from the `Deployment` folder** (i.e., `streamlit_app.py`, `requirements.txt`, `model.h5`) to this new repository.
3.  **Go to Streamlit Sharing** (share.streamlit.io) and log in with your GitHub account.
4.  **Click 'New app'** and select your newly created repository and the `streamlit_app.py` file.
5.  **Click 'Deploy!'** Streamlit will automatically install dependencies and launch your app.

Note: For larger models, consider storing `model.h5` on a cloud storage (e.g., Google Drive, Google Cloud Storage, Hugging Face Hub) and downloading it within your `streamlit_app.py` script to avoid GitHub file size limits.
""")
