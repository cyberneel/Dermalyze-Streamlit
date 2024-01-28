import streamlit as st
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from PIL import Image
from results import print_pred
from results import Predict
from io import BytesIO

st.set_page_config(
        page_title="Dermalyze",
        page_icon="images/avatar.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def save_file(uploaded_file):
    file_name = uploaded_file.name
    file_path = os.path.join("images", file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def crop(image):
    
    # Read the image
    image = cv2.imread(save_file(image))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Crop the image using the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)
    w+=100
    h+=100
    x-=50
    y-=50
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite('cropped_image.jpg', cropped_image)

def title():
    col1, mid, col2 = st.columns([1,1.62,20])
    with col1:
        st.image('images/avatar.png', width=78)
    with mid:
        st.write("    ")
    with col2:
        st.title("Dermalyze AI")
    st.subheader("Enter an image of the skin to analyze:")


title()

model = tf.keras.models.load_model('final_vgg1920epochs.h5', compile=True)

# Opening JSON file
f = open('dat.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

keys = list(data)


capimgs = st.camera_input('Take a picture')
    
    

if capimgs is not None:
    crop(image=capimgs)
    pred = Predict("cropped_image.jpg", keys, model, data, var=False)
        
    
    print_pred(pred)




    