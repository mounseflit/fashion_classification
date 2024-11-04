import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("model.h5")

# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale and resize to 28x28
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    return image

# Streamlit app interface
st.title("Fashion MNIST Clothing Classifier")
st.write("Upload an image of a clothing item, and the model will predict its category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the prediction
    st.write(f"*Prediction:* {predicted_class}")
    st.write("Confidence scores for each class:")
    for i, score in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {score:.2%}")
