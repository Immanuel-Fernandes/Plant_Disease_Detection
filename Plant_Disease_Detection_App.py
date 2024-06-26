# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Load the trained model
model = load_model('plant_disease_detection_model.h5')
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Function to preprocess the uploaded image and make predictions
def get_result(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    
    # Ensure the input shape matches the model's expectations
    assert x.shape == (1, 225, 225, 3), f"Expected input shape (1, 225, 225, 3), but got {x.shape}"
    
    predictions = model.predict(x)[0]
    return predictions

# Streamlit app layout
st.title("Plant Disease Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded file temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get predictions
        predictions = get_result("temp.jpg")
        predicted_label = labels[np.argmax(predictions)]

        # Display the prediction
        st.write(f"Prediction: {predicted_label}")

    except Exception as e:
        st.write("Error occurred: ", e)
