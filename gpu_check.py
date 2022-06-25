# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:01:32 2022

@author: neera
"""
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_built_with_cuda()
print(tf.version.VERSION)
import sys
sys.version
