# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:04:53 2022

@author: neera
"""

d_imgs1, d_imgs2 = model.predict([tf.expand_dims(tester[400], 0), tf.expand_dims(tester2[400], 0)])
print(d_imgs1.shape)
print(d_imgs2.shape)
#print(np.asarray(decoded_imgs).shape)
plt.subplot(221),
plt.imshow(d_imgs1[0][0])
plt.subplot(222),
plt.imshow(d_imgs2[0])
plt.subplot(223),
plt.imshow(tester[401][7])
plt.subplot(224),
plt.imshow(tester2[400])
