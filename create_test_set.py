# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:02:59 2022

@author: neera
"""

#loading the Signature dataset
path_test = "/kaggle/working/UCSD_Anomaly_Dataset_mod/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/"
dir_list = []
for i in range(1, 13, 1):
    dir_list.append(path_test + "Test" + str(i).zfill(3)+"/")
#dir_list = next(os.walk(path_train))[1]
#dir_list.sort()

#for training and testing data
tester = []
tester2 = []

#for storing the labels
test_labels = np.array([0]*1344)
test_labels[0*112 + 52: 112] = 1
test_labels[1*112 + 86 : 1*112 + 112] = 1
test_labels[2*112 + 0: 2*112+ 112] = 1
test_labels[3*112 + 22: 3*112+ 112] = 1
test_labels[4*112 + 0: 4*112+ 112] = 1
test_labels[5*112 + 0: 5*112+ 112] = 1
test_labels[6*112 + 37: 6*112+ 112] = 1
test_labels[7*112 + 0: 7*112+ 112] = 1
test_labels[8*112 + 0: 8*112+ 112] = 1
test_labels[9*112 + 0: 9*112+ 112] = 1
test_labels[10*112 + 0: 10*112+ 112] = 1
test_labels[11*112 + 79: 11*112+ 112] = 1
print(np.asarray(test_labels).shape)
    
#creating a mix of forged and genuine signatures to create a training and testing dataset
test_output = []
for path in dir_list:
    bunch = []
    for i in range(1, 121, 1):
        file_path = path + str(i).zfill(3)+".tif"
        temp = cv2.imread(file_path)
        temp = cv2.resize(temp, (128, 64))
        bunch.append(temp)
        if (i > 8):
            test_output.append(temp)
    for start in range(0, 112, 1):
        end = start + 8
        tester.append(bunch[start:end])
    print(np.asarray(tester).shape)
    

for image in tester:
    tester2.append(image[0])
    
tester2 = np.array(tester2)
print("tester2 shape: ", tester2.shape)
tester = np.array(tester)
test_output = np.array(test_output)
print(tester.shape)
print(test_output.shape)
plt.subplot(121)
plt.imshow(tester[1][7], cmap = 'gray')
plt.subplot(122)
plt.imshow(test_output[0],cmap = 'gray')

tester = tester.astype('float32') / 255.0
test_output = test_output.astype('float32') / 255.0
tester2 = tester2.astype('float32') / 255.0
