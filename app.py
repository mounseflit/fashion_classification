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
    
    # Apply a binary threshold for black background and white foreground
    threshold = 128  # Adjust the threshold if necessary
    image = image.point(lambda p: 0 if p < threshold else 255)  # Set background to black, foreground to white

    # Convert to NumPy array and normalize to [0, 1]
    image = np.array(image) / 255.0
    

    # Normalize the image and expand dimensions
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
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
