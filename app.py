import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('sports_model.h5')

# Define class names
class_names = ['basketball', 'horse racing', 'sumo wrestling', 'swimming', 'tennis']

# App title
st.title("Sports Image Classification")
st.write("Upload an image to classify the sport.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    img = image.resize((228, 228))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the result
    st.write(f"Prediction: {predicted_class}")
