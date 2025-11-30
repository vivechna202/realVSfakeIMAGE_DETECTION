import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# ============================
# FORCE ALL TEXT TO WHITE + BEAUTIFUL THEME
# ============================
# ============================
# FIXED & PERFECT CSS (JUST REPLACE YOUR OLD ONE)
# ============================
st.markdown("""
<style>
    /* Main dark background */
    .main, .stApp {
        background: linear-gradient(135deg, #0f0f23, #1a1a2e);
    }

    /* Neon glowing title */
    h1 {
        font-size: 3.8rem !important;
        font-weight: 900;
        text-align: center;
        color: white !important;
        text-shadow: 0 0 30px #00ff9d, 0 0 60px #00ff9d;
        margin-bottom: 10px !important;
    }

    /* All normal text = white */
    h2, h3, h4, h5, h6, p, div, span, .stMarkdown {
        color: white !important;
    }

    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 40px;
        opacity: 0.9;
    }

    /* ===== FIX 1: Make "Upload Image" & "Take a Photo" labels visible + pretty ===== */
    .stFileUploader > div > div > label,
    .stCameraInput > div > label {
        color: #00ff9d !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        text-align: center !important;
    }

    /* ===== FIX 2: Make "Drag and drop", "Limit 200MB...", "Clear photo" visible ===== */
    [data-testid="stFileUploaderDropzone"] div p,
    .css-1offfwp p,
    small,
    .uploadedFileName,
    button[kind="secondary"] {
        color: #000000 !important;
        font-size: 1rem !important;
    }

    /* ===== FIX 3: Show preview image inside camera/upload box again ===== */
    [data-testid="stFileUploader"] img,
    [data-testid="stCameraInput"] img,
    .stImage > img {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 255, 157, 0.3);
    }

    /* Result card (frosted glass) */
    .result-card {
        background: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(0,255,157,0.4);
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 40px rgba(0,255,157,0.15);
    }

    .big-text {font-size: 5rem !important; font-weight: 900; margin: 0;}
    .real {color: #00ff9d !important;}
    .fake {color: #ff4b4b !important;}
</style>
""", unsafe_allow_html=True)
# ============================
# TITLE
# ============================
st.markdown("<h1>Real vs Fake Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture any image — AI instantly detects Real or Fake</p>", unsafe_allow_html=True)

# ============================
# MODEL
# ============================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    url = "https://huggingface.co/Vivechna202/real-vs-fake-cnn/resolve/main/real_fake_model.keras"
    path = tf.keras.utils.get_file("real_fake_model.keras", origin=url)
    return tf.keras.models.load_model(path)

model = load_model()

# ============================
# INPUT
# ============================
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
with col2:
    camera = st.camera_input("Take a Photo")

image_bytes = uploaded.read() if uploaded else (camera.getvalue() if camera else None)

if image_bytes:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_input = cv2.resize(img, (240,240))
    img_input = (img_input.astype("float32")/255.0)[np.newaxis,...]
    
    with st.spinner("Analyzing..."):
        pred = float(model.predict(img_input, verbose=0)[0][0])
    
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1-pred

    st.markdown("---")
    colA, colB = st.columns([1.7, 1.3])
    
    with colA:
        st.image(img_rgb, use_column_width=True, caption="Your Image")
    
    with colB:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Result</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='big-text {label.lower()}'>{label.upper()}</p>", unsafe_allow_html=True)
        st.markdown(f"<h1>{confidence*100:.1f}%</h1>", unsafe_allow_html=True)
        st.progress(confidence)
        
        # Clean white-text chart
        fig, ax = plt.subplots(figsize=(5,3), facecolor='none')
        colors = ['#ff4b4b', '#00ff9d'] if label == "Real" else ['#00ff9d', '#ff4b4b']
        bars = ax.bar(["Fake", "Real"], [1-pred, pred], color=colors)
        ax.set_ylim(0,1)
        ax.set_facecolor('none')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{h:.1%}', 
                    ha='center', color='white', fontsize=14, fontweight='bold')
        plt.box(False)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align:center; opacity:0.8;'>Upload or capture an image to begin</h3>", unsafe_allow_html=True)
    st.balloons()