import streamlit as st
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
from PIL import Image

def print_pred(pred):
    st.markdown(f'''<u>**Problem:**</u> <br> <li>{pred[0].capitalize()}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**Solutions:**</u> <br> <li>{pred[5].capitalize()}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**Info:**</u> <br> <li>{pred[1]}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**Symptoms:**</u> <br> <li>{pred[2]}</li>''', unsafe_allow_html=True) 
    st.markdown(f'''<u>**Causes:**</u> <br> <li>{pred[3]}</li>''', unsafe_allow_html=True)
    st.markdown(f'''<u>**More Infor:**</u> <br> <li>{pred[4]}</li>''', unsafe_allow_html=True)

def Predict(uploaded_image, keys, model, data):
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

    return keys[prediction.argmax()],data[keys[prediction.argmax()]]['description'],data[keys[prediction.argmax()]]['symptoms'],data[keys[prediction.argmax()]]['causes'],data[keys[prediction.argmax()]]['treatement-1'],data[keys[prediction.argmax()]]['product']