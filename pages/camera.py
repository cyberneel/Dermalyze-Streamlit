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


title()

model = tf.keras.models.load_model('final_vgg1920epochs.h5', compile=True)

# Opening JSON file
f = open('dat.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

keys = list(data)

image = None
capimgs = st.camera_input('Take a picture')
if capimgs is not None:
    image = capimgs
    

if image is not None:
    pred = Predict(image, keys, model, data)
    
    print_pred(pred)




    