import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from utils import load_model, preprocess_image, predict_mask, postprocess_mask


# Load the model
MODEL_PATH = "./models/model_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(MODEL_PATH, DEVICE)


st.set_page_config(page_title="Let Me Detect - Retina", layout="centered")
st.title("🧠 Let Me Segmen - Retinal Segmentation")

uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Segmentation"):
        with st.spinner("Segmenting..."):
            input_tensor = preprocess_image(image).to(DEVICE)
            pred_mask = predict_mask(model, input_tensor)
            final_mask = postprocess_mask(pred_mask)

            st.image(final_mask, caption="Predicted Mask", use_column_width=True)
            st.success("Segmentation complete!")
