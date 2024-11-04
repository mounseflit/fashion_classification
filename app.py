import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the saved model with error handling
@st.cache_resource  # Caches the model so it doesn't reload on each run
def load_model():
    return tf.keras.models.load_model("model.h5")

# Attempt to load the model
try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"Error loading model: {e}")

# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)

    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to NumPy array
    image_array = np.array(image)

    # Determine the background color by checking the corners of the image
    corner_pixels = [
        image_array[0, 0],  # Top-left corner
        image_array[0, -1],  # Top-right corner
        image_array[-1, 0],  # Bottom-left corner
        image_array[-1, -1]  # Bottom-right corner
    ]
    background_color = np.mean(corner_pixels)  # Calculate average corner pixel brightness

    # If the background is light, invert the image to make the background dark
    if background_color > 128:  # Light background (white or light gray)
        image_array = 255 - image_array  # Invert colors

    # Normalize the image to [0, 1] for model compatibility
    image_array = image_array / 255.0

    # Expand dimensions to add batch and channel dimensions
    image_array = np.expand_dims(image_array, axis=(0, -1))
    
    return image_array


# Streamlit app interface
st.title("Fashion MNIST Clothing Classifier")
st.write("Upload an image of a clothing item, and the model will predict its category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None:
        # Preprocess and predict
        image = preprocess_image(image)
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]

        # Display the prediction
        st.write(f"*Prediction:* {predicted_class}")
        st.write("Confidence scores for each class:")
        for i, score in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {score:.2%}")
    else:
        st.write("Model is not loaded. Please ensure 'model.h5' is present in the project directory.")
