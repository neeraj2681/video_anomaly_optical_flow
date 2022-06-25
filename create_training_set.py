from two_image_flow import *
import numpy as np
import cv2

k = 8

#loading the training dataset
path_train = "/kaggle/working/video_anomaly_optical_flow/UCSD_Anomaly_Dataset_mod/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/"

#to store the path of every training folder
dir_list = []
for i in range(1, 17, 1):
  dir_list.append(path_train + "Train" + str(i).zfill(3)+"/")


#for training and testing data
train_input = []

#for output of training labels
train_prediction_label = []

#reading each train directory in list
for path in dir_list:

    #to collect the 'k' frames together
    bunch = []
    #iterating over each train image in respective training directory
    for i in range(1, 121, 1):
        #path of the current train frame
        file_path = path + str(i).zfill(3)+".tif"
        temp = cv2.imread(file_path)
        temp = cv2.resize(temp, (128, 64))
        bunch.append(temp)
        prev = temp
        #for each stack of frame put its next frame in the training set
        jar = []
        if (i > k):
            #train_prediction_output.append(temp)
            jar.append(get_optical_flow(prev, temp))
            jar.append(prev)
            train_prediction_label.append(jar)

    #crating a training set containing stack of 'k' frames
    for start in range(0, (120 - k), 1):
        end = start + k
        train_input.append(bunch[start:end])

#train_prediction_output = np.array(train_prediction_output)
train_prediction_label = np.array(train_prediction_label)
train_input = np.array(train_input)

np.save('train_set.npy',train_input)
#np.save('train_predcition_set', train_prediction_output)
np.save('train_prediction__set', train_prediction_label)


'''Checking the size of training sets'''
print('train_input_shape', (np.asarray(train_input)).shape)
#print('train_prediction_shape', np.asarray(train_prediction_output))
print('train_prediction_shape', (np.asarray(train_prediction_label)).shape)

#data augementation(beta phase)
'''black = [0, 0, 0]
for img in images:
  temp = cv2.imread(img)
  #constant= cv2.copyMakeBorder(temp, 1, 1, 1, 1,cv2.BORDER_CONSTANT,value=black)
  temp = cv2.resize(temp, (128, 64))

  trainer.append(temp)

#horizontal flip
for img in trainer[:1800]:
  temp = cv2.flip(img, 1)
  trainer.append(temp)

#vertical flip
for img in trainer[:1800]:
  temp = cv2.flip(img, 0)
  trainer.append(temp)

#vertical and horizontal flip
for img in trainer[:1800]:
  temp = cv2.flip(img, -1)
  trainer.append(temp)
'''
