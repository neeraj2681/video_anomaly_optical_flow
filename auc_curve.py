# -*- coding: utf-8 -*-
"""
Created on Sat May 21 19:05:34 2022

@author: neera
"""


test_scores = []
for i in range(0, len(tester)):
    print(i+1)
    mse = model.evaluate((tester[i:i+1], tester2[i:i+1]), (test_output[i:i+1], tester2[i:i+1]))
    test_scores.append(mse)

print(type(test_scores))
print(np.asarray(test_scores).shape)
print(test_labels.shape)
#print(test_scores)
#print("help: ", np.asarray(test_scores[0:1344, 0]).shape)
new_test_scores = [x[2] for x in test_scores]
print("printing: ", test_scores[1343][2])
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, new_test_scores)

from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr, label = None):
  plt.plot(fpr, tpr, linewidth = 2, label = label)
  plt.plot([0, 1], [0, 1], 'k--')

plot_roc_curve(fpr, tpr)
plt.show()

roc_auc_score(test_labels, new_test_scores)