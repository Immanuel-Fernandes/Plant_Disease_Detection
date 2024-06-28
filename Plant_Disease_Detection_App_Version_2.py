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

def display_prediction(image_path):
    # Display the uploaded image
    image = Image.open(image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Get predictions
    predictions = get_result(image_path)
    predicted_label = labels[np.argmax(predictions)]

    # Display the prediction
    st.write(f"Prediction: {predicted_label}")

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        display_prediction("temp.jpg")
    except Exception as e:
        st.write("Error occurred: ", e)

# Display sample images for demonstration purposes
st.subheader("Sample Images for Demonstration")

col1, col2, col3 = st.columns(3)

if col1.button('Upload Healthy Sample', key='button1'):
    display_prediction('sample_healthy.jpg')
with col1:
    st.image('sample_healthy.jpg', caption='Healthy', use_column_width=True)

if col2.button('Upload powdery Sample', key='button2'):
    display_prediction('sample_powdery.jpg')
with col2:
    st.image('sample_powdery.jpg', caption='Powdery Mildew', use_column_width=True)

if col3.button('Upload Rust Sample', key='button3'):
    display_prediction('sample_rust.jpg')
with col3:
    st.image('sample_rust.jpg', caption='Rust', use_column_width=True)
