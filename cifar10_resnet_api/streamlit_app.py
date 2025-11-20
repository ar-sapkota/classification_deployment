import streamlit as st
import requests
from PIL import Image
import io


# Configuration

API_URL = "http://fastapi:8001/predict"  # FastAPI backend URL

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title(" CIFAR-10 ResNet Image Classifier")

st.markdown(
    """
Upload an image and the model will predict which CIFAR-10 class it belongs to.
"""
)


# Upload image

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()

  
    # Send image to FastAPI
 
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {"file": ("image.jpg", byte_im, "image/jpeg")}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: **{result['class_name']}**")
                    st.info(f"Confidence: **{result['confidence']:.4f}**")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
