# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:03:43 2022

@author: neera
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from model import create_model

train_input = np.load('train_set.npy')
train_prediction = np.load('train_prediction_set.npy')
train_input = train_input / 255.0
train_prediction = train_prediction / 255.0

model = create_model()

history = model.fit(train_input, [train_input, train_prediction], batch_size = 1, epochs = 3, shuffle = False)
