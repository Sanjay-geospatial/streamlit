# 🌤️ Cloud-Free Image Generation with Pix2Pix GAN (TensorFlow)

This repository contains a **Jupyter Notebook** implementation of a **Pix2Pix GAN** model using **TensorFlow** for generating cloud-free satellite images from cloud-covered Sentinel imagery. The model is trained using data streamed directly from the cloud (Amazon Web Service), and the trained model is deployed in hugging face and also with a user-friendly **Streamlit web app**.

---

## 📌 Features

- ✅ Pix2Pix GAN model implemented using TensorFlow
- ✅ Uses Sentinel satellite images accessed from cloud storage (via STAC API)
- ✅ Fully interactive training in a Jupyter notebook (`pix2pix_training.ipynb`)
- ✅ Trained model saved and versioned in the Hugging face
- ✅ Streamlit app to visualize results and generate cloud-free images from new inputs

## 🧠 Model Training
- Sentinel-2 image pairs (cloudy and cloud-free) are downloaded from the cloud.

- Images are preprocessed and formatted as input-output training pairs.

- A Pix2Pix model (U-Net Generator + PatchGAN Discriminator) is trained.

- The trained model is saved to hugging face

  ## Streamlit app
- The app accepts only the image patches of size 128, 128, 3 as input
- Link to app : https://pix2pixcloudfree.streamlit.app/
