import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image

model = load_model(r"C:\Users\mwael\OneDrive\Desktop\home\course\male_female_detect\male_femal_detect.keras") # direct to save of model

st.title("Male vs Female Image Classifier")

st.write("Upload an image to classify:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img = np.array(img)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img = cv2.resize(img, (256, 256))
    
    img = img / 255.0
    img = np.reshape(img, (1, 256, 256, 3))
    
    pred = model.predict(img)
    output = pred[0][0]
    
    if output < 0.5:
        st.write("### The model predicts: **Male**")
    else:
        st.write("### The model predicts: **Female**")

