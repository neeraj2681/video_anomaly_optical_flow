import tensorflow as tf
from two_image_flow import get_optical_flow
import numpy as np
tf.config.run_functions_eagerly(True)

def prediction_loss(y_true, y_pred):
    error = 0.0
    for i in range(y_true.shape[0]):
        item = y_true[i]
        item_pred = y_pred[i]
        ground_truth_optflow = item[0]
        pred_opt_flow = get_optical_flow(item[1].numpy(), item_pred[0].numpy())
        temp_error = tf.reduce_mean(tf.math.square(pred_opt_flow - ground_truth_optflow), axis = -1)
        error = error + temp_error

    return (error / y_true.shape[0])
