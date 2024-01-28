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


def title():
    col1, mid, col2 = st.columns([1,1.62,20])
    with col1:
        st.image('images/avatar.png', width=78)
    with mid:
        st.write("    ")
    with col2:
        st.title("Dermalyze AI")
    st.subheader("Enter an image of the skin to analyze:")

st.set_page_config(
        page_title="Dermalyze",
        page_icon="\images\avatar.ico",
        layout="wide",
        initial_sidebar_state="expanded"
    )

title()

model = tf.keras.models.load_model('final_vgg1920epochs.h5', compile=True)

# Opening JSON file
f = open('dat.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

keys = list(data)



uploaded_image = st.file_uploader("Upload Image (jpg)")
img = None
if uploaded_image is not None:
    img = uploaded_image
    

if img is not None:
    pred = Predict(uploaded_image, keys, model, data)

    print_pred(pred)






    