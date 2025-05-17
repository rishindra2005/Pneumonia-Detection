import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import segmentation_models as sm
import cv2
import os

from flask import Flask, jsonify, request, send_file
# from flask_ngrok import run_with_ngrok
from io import BytesIO


app = Flask(__name__)

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
img_size = 448
# lung_test = preprocess_input(lungImg)

# def iou_loss(y_true, y_pred):
#     y_true = tf.reshape(y_true, [-1])
#     y_pred = tf.reshape(y_pred, [-1])
#     intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
#     score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + 
#     tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
#     return 1 - score

# def iou(y_true, y_pred):
#     y_pred = tf.round(tf.cast(y_pred, tf.int32))
#     intersect = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32), axis=[1])
#     union = tf.reduce_sum(tf.cast(y_true, tf.float32),axis=[1]) + tf.reduce_sum(tf.cast(y_pred, tf.float32),axis=[1])
#     k=10**-10
#     return tf.reduce_mean((intersect+k) / (union+k ))

from segmentation_models import Unet
model = Unet( BACKBONE,input_shape=(img_size,img_size,1), encoder_weights=None)
model.compile("Adam", loss=sm.losses.bce_jaccard_loss,metrics=[sm.metrics.iou_score],)

# Load Weights
model.load_weights("./infection_2dataset(resnet).hdf5")

def imagePred(imgPath):
    lungImg = []
    img_size = 448

    imgLung = cv2.imread(imgPath)
    imgLung = cv2.cvtColor(imgLung, cv2.COLOR_BGR2GRAY)

    imgLung = cv2.resize(imgLung, dsize = (img_size, img_size),interpolation = cv2.INTER_AREA).astype('float32')
    lungImg.append(imgLung[..., np.newaxis])
    lungImg = np.array(lungImg)
    lungImg[0][0][0]
    return lungImg
    

@app.route("/")
def hello():
    return "Hello World!! from anywhere in the world!"

@app.route("/predict",methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    fileUpload = request.files['file']
    path = os.path.join('./', fileUpload.filename)
    fileUpload.seek(0)
    fileUpload.save(path)
    def using_matplotlib():
        lungImg = imagePred(path)
        lung_test = preprocess_input(lungImg)
        predicted = model.predict(lung_test)
        fig = plt.figure(figsize = (15,15))

        plt.subplot(1,3,3)
        plt.imshow(lung_test[0], cmap = 'gray')
        plt.imshow(predicted[0],alpha = 0.5,cmap = "hot")
        plt.axis('off')

        img = BytesIO()
        plt.savefig(img, bbox_inches='tight', pad_inches=0, transparent=True)
        img.seek(0)
        return img
    strIO = using_matplotlib()
    os.remove(path)
    return send_file(strIO, mimetype='image/png')

if __name__ == "__main__":
  app.run()