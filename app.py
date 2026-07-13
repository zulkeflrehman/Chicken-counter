"""
Poultry Farm Chicken Counter (Streamlit version)
--------------------------------------------------
Detects and counts every visible chicken in an uploaded farm photo.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="Poultry Farm Chicken Counter", page_icon="🐔", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- Robust class detection ---------------------------------------------
# Auto-detects the "whole chicken" class instead of hardcoding a name,
# so this keeps working even if the model's class names change later.
EXCLUDE_KEYWORDS = ["head", "comb", "eye", "leg", "wing", "beak", "feather"]

def find_whole_chicken_class_id(names: dict) -> int:
    candidates = []
    for idx, name in names.items():
        lname = name.lower()
        if any(k in lname for k in ["chicken", "hen", "broiler", "poultry"]):
            if not any(kw in lname for kw in EXCLUDE_KEYWORDS):
                candidates.append((idx, name))
    if not candidates:
        return 0
    candidates.sort(key=lambda x: len(x[1]))
    return candidates[0][0]

CHICKEN_CLASS_ID = find_whole_chicken_class_id(model.names)

st.title("🐔 Poultry Farm Chicken Counter")
st.write("Upload a photo of a poultry farm and the model will detect and count every visible chicken. Supports PNG, JPG, JPEG, WEBP, and BMP.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Farm Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    confidence = st.slider(
        "Detection confidence threshold", 0.1, 0.9, 0.4, 0.05,
        help="Lower = catches more chickens but may include false positives. Higher = stricter, may miss some."
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting chickens..."):
        results = model(image, conf=confidence, verbose=False)

    boxes = results[0].boxes
    chicken_indices = [i for i, b in enumerate(boxes) if int(b.cls[0]) == CHICKEN_CLASS_ID]
    count = len(chicken_indices)

    # Draw only the whole-chicken boxes directly on a copy of the original
    # image. This avoids mutating Ultralytics' internal Results/Boxes
    # objects, which can produce corrupted output on some versions.
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)
    for i in chicken_indices:
        x1, y1, x2, y2 = [float(v) for v in boxes[i].xyxy[0].tolist()]
        conf = float(boxes[i].conf[0].item())
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{conf:.0%}", fill="lime")

    with col2:
        st.image(output_image, caption="Detection Result", use_column_width=True)

    if count == 0:
        st.warning("No chickens detected. Try a clearer or closer image, or lower the confidence threshold.")
    elif count == 1:
        st.success("We detected 1 chicken in the poultry farm.")
    else:
        st.success(f"We detected {count} chickens in the poultry farm.")

    if chicken_indices:
        avg_conf = float(np.mean([boxes[i].conf[0].item() for i in chicken_indices]))
        st.caption(f"Average detection confidence: {avg_conf:.0%}")
else:
    with col2:
        st.info("Upload an image to see results here.")

st.markdown("---")
st.caption(
    "This model was trained on a custom poultry-farm dataset. Accuracy may vary with "
    "image quality, lighting, and chicken density. For production use, verify counts "
    "against a manual sample before relying on results."
)
