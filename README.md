# Brain Tumor MRI Classification Streamlit App

## Project Overview
This project develops and deploys a Convolutional Neural Network (CNN) model to classify Brain MRI scans into two categories: 'Testing' (representing scans without a tumor, or a control group) and 'Training' (representing scans potentially containing a tumor, or the experimental group). The goal is to provide a deployable Streamlit application for easy inference.

## Data Loading and Preprocessing

1.  **Dataset Source:** The dataset is sourced from Kaggle (`masoudnickparvar/brain-tumor-mri-dataset`) using `kagglehub`.
2.  **Image Loading:** `tensorflow.keras.utils.image_dataset_from_directory` is used to efficiently load images from the dataset directory.
3.  **Splitting:** The dataset is split into training (80%) and validation (20%) sets.
4.  **Image Dimensions:** All images are resized to 128x128 pixels.
5.  **Batching:** Data is batched with a batch size of 32.
6.  **Performance Optimization:** Datasets are cached, shuffled, and prefetched for optimized training performance using `tf.data.AUTOTUNE`.

## Model Architecture and Training

1.  **Model Type:** A Sequential CNN model is built using TensorFlow/Keras.
2.  **Input Layer:** `Rescaling(1./255)` normalizes pixel values to the [0, 1] range.
3.  **Data Augmentation:** To prevent overfitting and improve generalization, the model incorporates `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.2)`, and `RandomZoom(0.2)` layers.
4.  **Convolutional Layers:** Three `Conv2D` layers with 16, 32, and 64 filters respectively, each followed by `MaxPooling2D` and `Dropout` (0.25).
5.  **Flattening:** The 2D feature maps are flattened into a 1D vector.
6.  **Dense Layers:** A `Dense` layer with 128 units and ReLU activation, followed by `Dropout` (0.5), and a final `Dense` layer with `softmax` activation for classification.
7.  **Compilation:** The model is compiled with the `adam` optimizer, `SparseCategoricalCrossentropy` loss function (since labels are integers), and `accuracy` as the metric.
8.  **Training:** The model is trained for 20 epochs using the training and validation datasets.

## Evaluation

After training, the model's performance is evaluated using:

-   **Accuracy and Loss Plots:** Training and validation accuracy/loss are plotted over epochs to visualize learning progression and identify overfitting.
-   **Quantitative Metrics:** A classification report is generated using `sklearn.metrics.classification_report` to show precision, recall, and F1-score for each class, along with overall accuracy.
-   **Qualitative Analysis:** A batch of validation images is displayed with their true and predicted labels to visually inspect the model's performance.

## Deployment Instructions

This folder contains all the necessary files to deploy the Brain Tumor MRI Classification model as a Streamlit web application.

### Files:
-   `streamlit_app.py`: The main Streamlit application script, which loads the trained model and provides an interface for image upload and prediction.
-   `requirements.txt`: Lists all Python dependencies required by the Streamlit app to run, including `streamlit`, `tensorflow`, `numpy`, `Pillow`, and `kagglehub`.
-   `model.h5`: The trained Keras deep learning model for image classification.

### How to Deploy on Streamlit Sharing:

1.  **Create a New GitHub Repository:** Initialize a new public or private GitHub repository (e.g., `brain-mri-classifier`).
2.  **Upload Files:** Add all files from this `Deployment` folder (`streamlit_app.py`, `requirements.txt`, `model.h5`) to the root of your new GitHub repository.
3.  **Go to Streamlit Sharing:** Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
4.  **Deploy the App:** Click on 'New app', select your newly created GitHub repository, and specify `streamlit_app.py` as the main application file.
5.  **Launch:** Click 'Deploy!' Streamlit will automatically install dependencies and launch your application, making it accessible via a public URL.

**Note:** For very large models (typically files larger than 100MB), consider storing `model.h5` on a cloud storage service (like Google Drive, Google Cloud Storage, or Hugging Face Hub). Then, modify `streamlit_app.py` to download the model at runtime from that cloud storage. This approach helps bypass GitHub's file size limits and optimizes repository management.
