# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:04:28 2022

@author: neera
"""

plt.plot(history.history["loss"], label="Training Loss")
print(history.history["loss"])
plt.plot(history.history["val_loss"], label="Validation Loss")
print(history.history["val_loss"])
plt.xlim([0, 10])
plt.ylim([0, 0.04])
plt.legend()