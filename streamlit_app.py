import streamlit as st
import tensorflow as tf
import numpy as np
import rioxarray as rxr
from huggingface_hub import hf_hub_download

# Load model using huggingface_hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="SanjayGeospatial/cloud-removal-model",
        filename="g_model_epoch1.h5"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# Preprocessing: Normalize and expand dims
def preprocess_image(img):
    img = (2 * img - img.min()) / (img.max() - img.min()) - 1
    return np.expand_dims(img.astype(np.float32), axis=0)

# Postprocessing: Scale prediction
def postprocess_image(pred):
    pred = ((pred + 1) / 2) ** 0.4
    return np.clip(pred.squeeze(), 0, 1)

# UI
st.title("☁️→🌤️ Cloud Removal App")
st.write("Upload a cloudy satellite image and get a cloud-free version!")

uploaded_file = st.file_uploader("Upload a GeoTIFF or image file", type=["tif", "tiff", "png", "jpg"])

if uploaded_file:
    input_raster = rxr.open_rasterio(uploaded_file)
    input_array = input_raster.transpose("y", "x", "band").values / 255.0

    st.image(np.clip(input_array, 0, 1), caption="🌥️ Input Image", use_column_width=True)

    with st.spinner("Generating cloud-free image..."):
        pred = model.predict(preprocess_image(input_array))
        output_image = postprocess_image(pred)

    st.image(output_image, caption="☀️ Output: Cloud-Free Image", use_column_width=True)
