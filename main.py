import streamlit as st
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from PIL import Image
from results import print_pred

if __name__ == '__main__':
    st.set_page_config(
        page_title="Dermalyze",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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



uploaded_image = st.file_uploader("Upload Image (jpg)")
img = None
if uploaded_image is not None:
    img = uploaded_image
    
    
def Predict(image):
    # Save the file to a directory
    with open(os.path.join("images", uploaded_image.name),"wb") as f:
        f.write(uploaded_image.getbuffer())
    st.success("Saved file: " + uploaded_image.name)

    # Load an image using PIL
    img = Image.open("images/" + uploaded_image.name)

    # Convert it to a numpy array
    img = np.array(img)

    img = cv2.resize(img, (32,32)) / 255.0
    prediction = model.predict(img.reshape(1,32,32,3))
    print(prediction)

    return keys[prediction.argmax()],data[keys[prediction.argmax()]]['description'],data[keys[prediction.argmax()]]['symptoms'],data[keys[prediction.argmax()]]['causes'],data[keys[prediction.argmax()]]['treatement-1']

if img is not None:
    pred = Predict(img)

    print_pred(pred)






    