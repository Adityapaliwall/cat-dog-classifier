import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image # Used to handle image files

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Cat & Dog Classifier",
    page_icon="🐾",
    layout="centered"
)


# --- MODEL LOADING ---
# Use st.cache_resource to load the model only once, which makes the app faster.
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from the .keras file.
    Includes a try-except block for robust error handling.
    """
try:
    import os
    from tensorflow.keras.models import load_model

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "cat_dog_classifier.keras")
    model = load_model(model_path)
    return model
except Exception as e:
    st.error(f"Error loading model: {e}")
    return None


# Load the model. The result is cached.
model = load_keras_model()


# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    """
    This function takes an uploaded image, preprocesses it to match the
    model's input requirements (64x64 pixels, normalized).
    """
    # Resize the image to 64x64 pixels, as required by the model
    img = image.resize((64, 64))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # The model expects a "batch" of images, so we add a new dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    return img_array


# --- STREAMLIT APP INTERFACE ---
st.title("🐾 Cat & Dog Image Classifier")

st.write(
    "Upload an image and our trained neural network will predict whether it's a cat or a dog. "
    "This model was trained on thousands of images to learn the features of each animal."
)

# File uploader widget that accepts common image formats
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the image that the user uploaded
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Image', use_column_width=True)
    st.write("")
    st.info("Classifying...")

    # Preprocess the image and get the prediction from the model
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    # The model's output is a probability score
    probability = prediction[0][0]

    # Display the final prediction and confidence score
    if probability >= 0.5:
        st.success(f"**Prediction: It's a Dog! 🐕**")
        st.write(f"**Confidence:** `{probability*100:.2f}%`")
    else:
        st.success(f"**Prediction: It's a Cat! 🐈**")
        st.write(f"**Confidence:** `{(1 - probability)*100:.2f}%`")

elif model is None:
    st.error("Model file not found. Please ensure 'cat_dog_classifier.keras' is in the same directory as app.py.")

st.sidebar.write("---")
st.sidebar.write("**About this App**")

st.sidebar.write("This app uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras.")

