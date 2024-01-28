import json
from keras.models import Model, load_model
import gradio as gr
import cv2


model = load_model('final_vgg1920epochs.h5', compile=True)

# Opening JSON file
f = open('dat.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)

keys = list(data)


def Predict(image):
  img = cv2.resize(image, (32,32)) / 255.0
  prediction = model.predict(img.reshape(1,32,32,3))
  print(prediction)
  
  
  return keys[prediction.argmax()],data[keys[prediction.argmax()]]['description'],data[keys[prediction.argmax()]]['symptoms'],data[keys[prediction.argmax()]]['causes'],data[keys[prediction.argmax()]]['treatement-1']

demo=gr.Interface(fn=Predict,
                  inputs="image",
                  outputs=[gr.inputs.Textbox(label='Name Of Disease'),gr.inputs.Textbox(label='Description'),gr.inputs.Textbox(label='Symptoms'),gr.inputs.Textbox(label='Causes'),gr.inputs.Textbox(label='Treatement')],
                  title="Skin Disease Classification",
                  description='This Space predict these disease:\n \n1) Acne and Rosacea Photos. \n2) Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions.\n3) Eczema Photos. \n4) Melanoma Skin Cancer Nevi and Moles.\n5) Psoriasis pictures Lichen Planus and related diseases.\n6) Tinea Ringworm Candidiasis and other Fungal Infections.\n7) Urticaria Hives.\n8) Nail Fungus and other Nail Disease.\n')

demo.launch(debug=True)