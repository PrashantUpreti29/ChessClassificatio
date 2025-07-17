import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("chess_model.h5")

# Class labels
class_labels = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
img_size = (128, 128)

# Title
st.title("♟️ Chess Piece Classifier")
st.write("Upload an image of a chess piece to classify it.")

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption='Uploaded Image', use_column_width=True)

# Preprocess
img = img.resize(img_size)
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Display
st.success(f"Prediction: {predicted_class} ({confidence:.2f}%)")
st.write("Class Probabilities:")
for label, prob in zip(class_labels, prediction[0]):
st.write(f"{label}: {prob*100:.2f}%")
