import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Poultry Counter AI",
    page_icon="🐓",
    layout="centered", # Centered layout keeps everything tightly organized
    initial_sidebar_state="collapsed"
)

# 2. Premium UI Styling
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f9fbf9; }
    
    /* Clean, modern card for the counter */
    .counter-card {
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    .counter-title { font-size: 16px; opacity: 0.9; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
    .counter-value { font-size: 48px; font-weight: bold; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# 3. Cached Model Loading (Prevents app lag)
@st.cache_resource
def load_model():
    return YOLO("chicken_detector_best.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load 'chicken_detector_best.pt'. Please ensure the file is in this folder.")
    st.stop()

# 4. App Header
st.title("🐓 Poultry Farm Chicken Counter")
st.markdown("Instantly count your flock using high-accuracy AI vision.")

# 5. Dropzone File Uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# 6. App Logic & Display
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Run AI Detection with a clean loading spinner
    with st.spinner("Analyzing farm photo..."):
        results = model(np.array(image), conf=0.4)[0]
        annotated_img = results.plot()
        chicken_count = len(results.boxes)

    # Big, Bold Beautiful Counter Card
    st.markdown(f"""
        <div class="counter-card">
            <div class="counter-title">🐣 Total Chickens Detected</div>
            <div class="counter-value">{chicken_count}</div>
        </div>
    """, unsafe_allow_html=True)

    # Clean, full-width result image
    st.image(annotated_img, caption="AI Analysis Result", use_container_width=True)

else:
    # Minimalist, inviting welcome state
    st.info("👋 Upload a farm photo above to get an instant count.")
