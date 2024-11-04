# Fashion MNIST Clothing Classifier

This is a simple web application built with Streamlit and TensorFlow that classifies images of clothing items into one of the Fashion MNIST categories. Upload an image of a clothing item, and the model will predict its category, such as "T-shirt/top", "Trouser", "Pullover", and more.

## How It Works

1. **Upload an image**: The app takes an image as input.
2. **Image Processing**: The image is converted to grayscale, resized to 28x28 pixels, and normalized.
3. **Prediction**: The pre-trained model classifies the image into one of the Fashion MNIST categories.
4. **Display**: The app shows the predicted category along with confidence scores for each class.

## Prerequisites

- Python 3.7 or above

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/mounseflit/fashion_classification.git
   cd fashion_classification
