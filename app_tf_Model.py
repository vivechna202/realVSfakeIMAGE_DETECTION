import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt

# -------------------- UI PAGE CONFIG --------------------
st.set_page_config(page_title="Real vs Fake Image Detector", layout="centered")

# -------------------- CUSTOM CSS ------------------------
st.markdown("""
<style>
    .title { 
        text-align: center; 
        font-size: 38px; 
        font-weight: 800; 
        color: #2c3e50; 
    }
    .prediction-box {
        padding: 18px;
        border-radius: 14px;
        background-color: #eef2f3;
        text-align: center;
        font-size: 22px;
        font-weight: 650;
        margin-top: 10px;
        border: 2px solid #d0d7de;
    }
    .css-1d391kg, .css-18e3th9 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE SECTION ---------------------
st.markdown("<div class='title'>🕵️ Real vs Fake Image Detection</div>", unsafe_allow_html=True)
st.write("Upload an image or capture from webcam to check if it's Real or Fake.")

# -------------------- SIDEBAR INFO ----------------------
with st.sidebar:
    st.header("ℹ️ About this Project")
    st.write("This system detects whether an image is Real or Fake using a CNN model converted to TensorFlow Lite.")
    st.write("Built using **Streamlit, TensorFlow Lite, OpenCV & Matplotlib**.")
    st.write("Upload or capture an image to get prediction with confidence score & probability chart.")
    st.write("---")
    st.info("👨‍🎓 Developer: Vivechana Singh\n🧠 Model: TFLite Converted CNN")


# ------------------- LOAD TFLITE MODEL ------------------
interpreter = tf.lite.Interpreter(model_path=r"cifake_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Fake", "Real"]

# -------------------- IMAGE INPUT -----------------------
uploaded = st.file_uploader("📂 Drag & Drop or Select an Image", type=["jpg", "png", "jpeg"])
camera = st.camera_input("📸 Capture from Webcam")

image_source = None
if uploaded:
    image_source = uploaded.read()
elif camera:
    image_source = camera.getvalue()

# -------------------- PROCESS & PREDICT -----------------
if image_source:

    file_bytes = np.frombuffer(image_source, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_pred = cv2.resize(img, (240, 240))
    img_pred = img_pred.astype(np.float32) / 255.0

    # PROGRESS BAR
    progress_bar = st.progress(0)
    for pct in range(100):
        time.sleep(0.01)
        progress_bar.progress(pct + 1)

    with st.spinner("🔍 Analyzing image..."):
        input_data = np.expand_dims(img_pred, 0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Real" if pred > 0.5 else "Fake"
    confidence = round(float(pred if pred > 0.5 else 1 - pred), 3)

    # ------------------- TABS UI -------------------------
    tab1, tab2 = st.tabs(["📷 Image Preview", "📊 Prediction"])

    with tab1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    with tab2:
        st.write("### 🧠 Prediction Result")

        if label == "Real":
            st.success("🟢 REAL IMAGE")
        else:
            st.error("🔴 FAKE IMAGE")

        st.markdown(f"<div class='prediction-box'>Confidence: {confidence*100:.1f}%</div>", unsafe_allow_html=True)

        # Chart
        st.write("### Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], [1 - pred, pred])
        st.pyplot(fig)

        st.download_button("📥 Download Result", f"Prediction: {label}\nConfidence: {confidence*100:.1f}%")


# ---------------------------------------------------------------------
else:
    st.warning("Please upload an image or capture from webcam to continue.")
