'''Defining an autoencoder with a single volumetric input branch,
and two branch output model.
1. First output branch reconstructs the input volume(stack of frames)
2. Second output branch predicts next frame wrt input stack of frames
3. MSE is used for reconstruction branch
4. Optical flow is calculated for predicted branch output and based on that MSE is calculated'''

from two_image_flow import *
import tensorflow as tf
from custom_loss import prediction_loss
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
tf.compat.v1.enable_eager_execution()

#from custom_loss import *

def create_model():


    input_init_3d = layers.Input(shape=(8, 64, 128, 3))
    el_3d1 = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(input_init_3d)
    el_3d1_normalized = tf.keras.layers.BatchNormalization()(el_3d1)
    el_3d1_activated = tf.keras.layers.Activation("selu")(el_3d1_normalized)
    el_3d1_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d1_activated)
    el_3d2 = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(el_3d1_pooled)
    el_3d2_normalized = tf.keras.layers.BatchNormalization()(el_3d2)
    el_3d2_activated = tf.keras.layers.Activation("selu")(el_3d2_normalized)
    el_3d2_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d2_activated)
    el_3d3 = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(el_3d2_pooled)
    el_3d3_normalized = tf.keras.layers.BatchNormalization()(el_3d3)
    el_3d3_activated = tf.keras.layers.Activation("selu")(el_3d3_normalized)
    el_3d3_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d3_activated)
    #layers.Conv2D(512, (3, 3), padding = 'same'),
    #keras.layers.BatchNormalization(),
    #keras.layers.Activation("selu"),
    #layers.MaxPool2D(pool_size = 2),'''

    #Concatenation layer
    '''
    d3_latenet_mod_input = layers.Reshape((8, 16, 128))(el_3d3_pooled)
    concat = layers.concatenate([el_2d3_pooled, d3_latenet_mod_input], axis = 3)
    d3_latent_mod_output = layers.Reshape((1, 8, 16, 256))(concat)
    '''

    '''Reconstruction branch'''
    dl_3d1_r = layers.Conv3DTranspose(128, kernel_size = (3, 3, 3), padding = "same", strides = (2, 2, 2))(el_3d3_pooled)
    dl_3d1_normalized_r = tf.keras.layers.BatchNormalization()(dl_3d1_r)
    dl_3d1_activated_r = tf.keras.layers.Activation("selu")(dl_3d1_normalized_r)
    dl_3d2_r = layers.Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), padding='same', strides = (2, 2, 2))(dl_3d1_activated_r)
    dl_3d2_normalized_r = tf.keras.layers.BatchNormalization()(dl_3d2_r)
    dl_3d2_activated_r = tf.keras.layers.Activation("selu")(dl_3d2_normalized_r)
    dl_3d3_r = layers.Conv3DTranspose(32, kernel_size = (3, 3, 3), padding = "same", strides = (2, 2, 2))(dl_3d2_activated_r)
    dl_3d3_normalized_r = tf.keras.layers.BatchNormalization()(dl_3d3_r)
    dl_3d3_activated_r = tf.keras.layers.Activation("selu")(dl_3d3_normalized_r)
    dl_3d_reconstruction_r = layers.Conv3D(3, kernel_size = (3, 3, 3), strides = 1, activation = "sigmoid", padding = "same", name = 'reconstruction')(dl_3d3_activated_r)


    '''Prediction branch'''
    dl_3d1 = layers.Conv3DTranspose(128, kernel_size = (3, 3, 3), padding = "same", strides = (1, 2, 2))(el_3d3_pooled)
    dl_3d1_normalized = tf.keras.layers.BatchNormalization()(dl_3d1)
    dl_3d1_activated = tf.keras.layers.Activation("selu")(dl_3d1_normalized)
    dl_3d2 = layers.Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), padding='same', strides = (1, 2, 2))(dl_3d1_activated)
    dl_3d2_normalized = tf.keras.layers.BatchNormalization()(dl_3d2)
    dl_3d2_activated = tf.keras.layers.Activation("selu")(dl_3d2_normalized)
    dl_3d3 = layers.Conv3DTranspose(32, kernel_size = (3, 3, 3), padding = "same", strides = (1, 2, 2))(dl_3d2_activated)
    dl_3d3_normalized = tf.keras.layers.BatchNormalization()(dl_3d3)
    dl_3d3_activated = tf.keras.layers.Activation("selu")(dl_3d3_normalized)
    dl_3d_prediction = tf.keras.layers.Conv3D(3, kernel_size = (3, 3, 3), strides = 1, activation = "sigmoid", padding = "same", name = 'prediction')(dl_3d3_activated)

    print("check for eager mode:", tf.executing_eagerly())

    model = tf.keras.Model(inputs = input_init_3d, outputs = [dl_3d_reconstruction_r, dl_3d_prediction])
    model.compile(optimizer = 'adam', loss={'reconstruction': 'mse', 'prediction': prediction_loss}, loss_weights=[0.5, 0.5])

    model.summary()
    return model
