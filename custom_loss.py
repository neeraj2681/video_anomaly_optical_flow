import tensorflow as tf
from two_image_flow import get_optical_flow
import numpy as np
tf.config.run_functions_eagerly(True)

def prediction_loss(y_true, y_pred):
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    error = 0.0
    for i in range(y_true.shape[0]):
        item = y_true[i]
        item_pred = y_pred[i]
        ground_truth_optflow = item[0]
        print("item[0] shape and type:", item[0].shape, type(item[0]))
        print("item[1] shape and type:", item[1].shape, type(item[1]))
        pred_opt_flow = get_optical_flow(item[1].numpy(), item_pred[0].numpy())
        error = error + tf.reduce_mean(tf.math.square(pred_opt_flow - ground_truth_optflow), axis = -1)

    return error / y_true.shape[0]
