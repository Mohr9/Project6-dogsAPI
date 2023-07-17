
##### Importation des packaging : #####

import os
import pickle

import streamlit as st
import base64

import numpy as np
import pandas as pd
from io import BytesIO

from PIL import Image
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###### Chargement des données ######

#Chargement des modèles : 
vgg_model =  tf.keras.models.load_model("vgg16_trained.h5") 
inception_model = tf.keras.models.load_model("inception_trained.h5") 
xception_model = tf.keras.models.load_model("xception_trained.h5") 

#Chargement des labels : 
label = pickle.load(open('encoder','rb'))




 ###### Importation des fonctions  ######
 
 # egalisation des  histogrammes : 
def equalize_histogram(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape)
    return equalized_image.astype(np.uint8)

# Normalisation des images
def normalize_image(image):
    return image / 255.0

# Blanchiment des images
def whiten_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def preprocess_image(image):
    image = equalize_histogram(image)
    image = normalize_image(image)
    image = whiten_image(image)

    return image


#Fonction prédiction :

def tf_predict_from_image(image, model):
    # Convertir l'image en couleur
    image_rgb = image.convert('RGB')

    # Redimensionnement de l'image pour la mettre en format pixel (300,300)
    image_rgb = image_rgb.resize((300, 300))

    # Convertir l'image en tableau numpy
    test_image = np.array(image_rgb)

    # Rajout d'une dimension pour que l'image soit compatible avec les modèles de Transfer Learning :
    # la dimension sera sous forme : (batch_size, height, width, channels)
    test_image = np.expand_dims(test_image, axis=0)

    # Prétraitement de l'image :
    test_image = preprocess_image(test_image)

    # Prédiction avec le modèle choisi :
    prediction = model.predict(test_image)

    # Indice de la classe prédite
    predicted_class_index = np.argmax(prediction)

    # Nom de la classe prédite
    predicted_class_label = label[predicted_class_index]

    # Retourner la prédiction
    return predicted_class_label


    
    ###### Mise en page #######
    
    
 ######## Fond d'écran : 
 
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.jpg')
    
    
###### selection de modèle :
st.title("Sélectionnez un modèle de prédiction :")
    # Liste des modèles disponibles
models = {
    'InceptionV3': inception_model,
    'Xception': xception_model,
     'VGG16': vgg_model,

}

selected_model = st.selectbox("Choisissez un modèle", list(models.keys()))

# Chargement du modèle sélectionné
st.write(f"Vous avez selectionné le modèle {selected_model}")

#Récupération du modèle utilisé : 
model_selected = models[selected_model]


####### pour l'upload d'image :

st.title("Prédiction de race de chien à partir d'une image :")

# Interface utilisateur pour télécharger une image
uploaded_file = st.file_uploader("Veuillez uploader votre image (format accepté : jpg, jpeg, png) ", type=['jpg', 'jpeg', 'png'])

# Vérification si une image a été chargée

if uploaded_file is not None: #si l'utilisateur a bien chargée une image
    # Lecture de l'image téléchargée 
    image = Image.open(uploaded_file)
    # Affichage de l'image téléchargée
    st.image(image, caption='Image téléchargée avec succès !')

    # Prédiction avec le modèle choisi
    predicted_class = tf_predict_from_image(image, model_selected)
    
    # Afficher la prédiction
    st.write("Prédiction de la race de chien :", predicted_class)
    
else:
    st.warning("Merci d'uploader une image valide.")






