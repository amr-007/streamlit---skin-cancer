import streamlit as st
import tensorflow as tf
# import the necessary packages for image recognition
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib

# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon="hair-bulb-skin-layers-follicle-epidermis-flat-vector-30077518.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    )
st.title("Skin Cancer Diagnosis")
st.markdown('This project aims to help dermatologists to diagnose skin lesions images and determines wether it is cancerous or not.')
st.markdown('This is done using Convolutional Neural Network.')
st.sidebar.subheader("Input")

models_list = ["Xception", "EffecientNetB5", "VGG19"]
network = st.sidebar.selectbox("Select the Model", models_list)

@st.cache(allow_output_mutation=True)
def get_model(model_name):
    load = tf.keras.models.load_model(model_name)
    return (load)

model_1 = get_model('Xcep_multi290_new.hdf5')
model_2 = get_model('EFNETB5_model.hdf5')
model_3 = get_model('VGG_model.hdf5')

# define a dictionary that maps model names to their classes

MODELS = {
    "Xception": model_1,
    "EffecientNetB5": model_2,
    "VGG19": model_3
}



uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg"]
)


if uploaded_file:
    bytes_data = uploaded_file.read()

    inputShape = (224, 224)
    preprocess = preprocess_input

    if network in ("Xception"):
        inputShape = (290, 290)
        preprocess = preprocess_input

    model = MODELS[network]


    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image /= 255.


    preds = model.predict(image)

    class_names = {0: 'Melanocytic Nevi',
                   1: 'Basal Cell Carcinoma',
                   2: 'Actinic Keratosis',
                   3: 'Benign Keratosis',
                   4: 'Dermatofibroma',
                   5: 'Vascular Lesion',
                   6: 'Squamous Cell Carcinoma ',
                   7: 'Melanoma' }    
    index = np.argmax(preds) 
    class_list = ['Melanocytic Nevi', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion', 'Squamous Cell Carcinoma ', 'Melanoma']
    st.image(bytes_data, width=600)
    s = pd.Series(preds[0],index=class_list)
    df = pd.DataFrame(s, columns=['confidence'])   
    
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    def callback():
        st.session_state.button_clicked = True 

    if (
        st.button('predict', on_click=callback)
        or st.session_state.button_clicked
    ):
      st.title("Prediction - {}".format(class_names[index])) 

    
if st.sidebar.checkbox("show top 5 predictions"):
    st.subheader(f"Top Predictions from {network} model")
    st.dataframe(
        df.sort_values(by=['confidence'], ascending=False).head(5)
        )

if st.sidebar.checkbox("Show classification report"):

    if network in ("Xception"):
        st.image('Xcep_metrics.PNG', width=700, caption='Xception model classification report')

    elif network in ("EffecientNetB5"):
        st.image('efn_metrics.PNG', width=700, caption='EffecientNetB5 model classification report')

    else:
        st.image('VGG_metrics.PNG', width=700, caption='VGG19 model classification report')


if st.sidebar.checkbox("Show confusion matrix"):

    if network in ("Xception"):
        st.image('Xcep_cnf.png', width=700, caption='Xception model Confusion Matrix')

    elif network in ("EffecientNetB5"):
        st.image('efn_cnf.png', width=700, caption='EffecientNetB5 model Confusion Matrix')

    else:
        st.image('VGG_cnf.png', width=700, caption='VGG19 model Confusion Matrix')

