import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real vs Fake Image Detection", layout="centered")

st.title("🕵️ Real vs Fake Image Detection using CNN")

model = tf.keras.models.load_model(r"C:\Users\Vivechna Singh\Downloads\real_fake_model.keras")  
class_names = ["Fake", "Real"]

st.write("### Upload an image or capture from webcam")

uploaded = st.file_uploader("📂 Drag & Drop or Select an Image", type=["jpg","png","jpeg"])
camera = st.camera_input("📸 Capture from Webcam")

image_source = None

if uploaded:
    image_source = uploaded.read()
elif camera:
    image_source = camera.getvalue()

if image_source:

    # Read original image for display
    file_bytes = np.frombuffer(image_source, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Prepare resized image for prediction
    img_pred = cv2.resize(img, (240,240))
    img_pred = img_pred.astype(np.float32) / 255.0

    # Loading spinner
    with st.spinner("🔍 Analyzing image... Please wait"):
        time.sleep(1.5)  # simulate processing time
        pred = model.predict(np.expand_dims(img_pred, 0))[0][0]

    label = "Real" if pred > 0.5 else "Fake"
    confidence = round(float(pred if pred > 0.5 else 1-pred), 3)

    col1, col2 = st.columns(2)

    # Display Image
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    # Display Results
    with col2:
        st.write("### 🧠 Prediction Result")
        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.1f}%")

        # Probability bar chart
        st.write("### 📊 Probability Chart")
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], [1-pred, pred])
        st.pyplot(fig)
