import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Poultry Farm Chicken Counter")

model = YOLO("chicken_detector_best.pt")

uploaded_file = st.file_uploader("Upload a farm photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model(np.array(image), conf=0.4)[0]
    annotated = results.plot()

    st.image(annotated, caption="Detected chickens", use_column_width=True)
    st.success(f"Total chickens detected: {len(results.boxes)}")
