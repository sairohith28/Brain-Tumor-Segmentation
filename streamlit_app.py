# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO

# Loading the Model
model = load_model('./model.h5', compile=False)

st.markdown("## Brain Tumor Segmentation App")
st.markdown(""" This app accepts brain MRI images and returns segmented tumor predictions

**Made by Sai Rohith**

""")

st.markdown("Upload a brain MRI image")

# Uploading the image
mri_image = st.file_uploader("Upload an image...", type=['png', 'jpg', 'webp'])
submit = st.button('Predict')

# On predict button click
if submit:
    if mri_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(mri_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Resize the image to match the expected input shape of the model
        resized_image = cv2.resize(opencv_image, (256, 256))
        resized_image = np.expand_dims(resized_image, axis=0)

        # Preprocess the image (if required) before feeding it to the model

        # Normalize pixel values to the range [0, 1]
        resized_image = resized_image / 255.0

        # Perform prediction
        predictions = model.predict(resized_image)

        # Convert predictions to binary mask
        mask = (predictions > 0.5).astype(np.uint8)[0] * 255

        # Resize the mask to match the dimensions of the input image
        mask = cv2.resize(mask, (opencv_image.shape[1], opencv_image.shape[0]))

        # Convert the mask to the same data type as the input image
        mask = mask.astype(np.uint8)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(opencv_image, opencv_image, mask=mask)

        # Display the images side by side
        col1, col2, col3 = st.columns(3)
        col1.image(opencv_image, channels="BGR", caption='Original Image')
        col2.image(mask, caption='Segmentation Mask')
        # col3.image(masked_image, channels="BGR", caption='Masked Image')
