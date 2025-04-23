import streamlit as st
import tensorflow as tf
import numpy as np
import rioxarray as rxr
from huggingface_hub import hf_hub_download
import folium
import rasterio
from streamlit_folium import st_folium
from folium.raster_layers import ImageOverlay
from branca.colormap import linear

# Load model using huggingface_hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="SanjayGeospatial/cloud-removal-model",
        filename="g_model_epoch1.h5",
        force_download=True
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

st.title("☁️→🌤️ Cloud Removal App")
st.write("Upload a cloudy satellite image and get a cloud-free version!")

uploaded_file = st.file_uploader("Upload a GeoTIFF or image file", type=["tif", "tiff"])

if uploaded_file:
    # Read and preprocess
    input_image = rxr.open_rasterio(uploaded_file)
    input_image = input_image.transpose('y', 'x', 'band').data
    image_np = np.array(input_image)
    
    with st.spinner("Generating cloud-free image..."):
        input_tensor = preprocess_image(image_np)
        output_tensor = model.predict(input_tensor)
        output_image = postprocess_image(output_tensor[0])
    
    # Normalize to 0-255 for visualization
    input_vis = ((image_np + 1) / 2)**0.4
    output_vis = ((output_image + 1) / 2)**0.4
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(input_vis, caption="☁️ Input: Cloudy Image", use_container_width=True, clamp = True)
    
    with col2:
        st.image(output_vis, caption="🌤️ Output: Cloud-Free Image", use_container_width=True, clamp =True)
        
