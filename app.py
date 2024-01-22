import streamlit as st
import tensorflow.compat.v2 as tf
import keras.utils as image
import numpy as np
from PIL import Image, ImageOps  # Streamlit works with PIL library very easily for Images
import cv2
#import utils.SQLiteDB as dbHandler
#from app import prediction
import os

# Crop Model

import streamlit as st
import joblib

# Load the trained model
model_path = 'model.joblib'
RF = joblib.load(model_path)

def main():
    st.title("Crop Recommendation App")

    # User input features
    N, P, K, temperature, humidity, ph, rainfall = get_user_input()

    # Make prediction
    if st.button("Predict", key='Corp'):
        result = make_prediction(RF, N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"The recommended crop is {result}")

def get_user_input():
    st.sidebar.header("User Input Features")

    N = st.sidebar.text_input("Nitrogen (N)", 50)
    P = st.sidebar.text_input("Phosphorus (P)", 50)
    K = st.sidebar.text_input("Potassium (K)", 50)
    temperature = st.sidebar.text_input("Temperature", 25.0)
    humidity = st.sidebar.text_input("Humidity", 50)
    ph = st.sidebar.text_input("pH", 7.0)
    rainfall = st.sidebar.text_input("Rainfall", 50.0)

    return N, P, K, temperature, humidity, ph, rainfall


def make_prediction(RF, N, P, K, temperature, humidity, ph, rainfall):
    new_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = RF.predict(new_data)
    return prediction[0]

if __name__ == "__main__":
    main()




#Crop file ends




#path = 'C:\\Users\\yashpal kharpuriya\\PycharmProjects\\PythonProject5\\upload'
model_path = 'C:\\Users\\yashpal kharpuriya\\PycharmProjects\\PythonProject5\\PestImageClassificationInception.h5'
#model_path = 'C:\\Users\\asus\\PycharmProjects\\PestClassification\\PestImageClassificationCNN.h5'

def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


st.title("Pest Image Classification using CNN")
upload = st.file_uploader('Upload a pest image')

def prediction(savedModel, inputImage):
    test_image = image.load_img( inputImage, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = savedModel.predict(test_image)
    print("Predicted result", result)
    return result




if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Color from BGR to RGB
    print("type of opencv", type(opencv_image))
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)
    if st.button('Predict Pest', key='Pest'):
        # Load pretrained Model
        model = tf.keras.models.load_model(model_path)

        path_dir = os.path.join(os.getcwd(), 'upload')
        print("path_dir =", path_dir)
        upload_path = os.path.join(path_dir, upload.name)
        print("upload_path=", upload_path)

        # Save uploaded file
        save_uploadedfile(upload, upload_path)

        # Prediction on uploaded image
        result = prediction(model, upload_path)
        #map_result = {1: 'beetle', 3: 'Black hairy', 0: 'corn earworm', 4: 'Field Cricket', 2: 'Termite'}//CNN
        map_result = {3:'beetle',0:'Black hairy',4:'corn earworm',1:'Field Cricket',2:'Termite'} #INCEPTION
        print('np array',np.argmax(result))
        print("Predicted result", result)
        print("Output labels = ", map_result)
        print("output[np.argmax(result)] = ", map_result[np.argmax(result)])
        st.title( map_result[np.argmax(result)])
        detail_result = {3:'This is a beetle',0:'This is a Black hairy',4:'This is a corn earworm',1:'This is a Field Cricket',2:'This is a Termite'}
        st.text(detail_result[np.argmax(result)])


        #Edit
        def main():
            st.title("Smart Agriculture App")

            # Select the task
            task = st.sidebar.selectbox("Select Task", ["Crop Recommendation", "Pest Image Classification"])

            if task == "Crop Recommendation":
                predict_crop()
            elif task == "Pest Image Classification":
                upload = st.file_uploader('Upload a pest image')
                predict_pest(upload)
