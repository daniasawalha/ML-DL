# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:46:23 2023

@author: Dania
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
n_samples = 50
data, labels = make_blobs(n_samples=n_samples,
centers=([1.1, 3], [4.5, 6.9]),
random_state=0)
colours = ('green', 'orange')
fig, ax = plt.subplots()
for n_class in range(2):
 ax.scatter(data[labels==n_class][:, 0],
 data[labels==n_class][:, 1],
 c=colours[n_class],
 s=50,
 label=str(n_class))

from sklearn.model_selection import train_test_split
datasets = train_test_split(data,
labels,
test_size=0.2)
train_data, test_data, train_labels, test_labels = datasets
from sklearn.linear_model import Perceptron
p = Perceptron(random_state=42)
p.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score
predictions_train = p.predict(train_data)
predictions_test = p.predict(test_data)
train_score = accuracy_score(predictions_train, train_labels)
print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)
p.score(train_data, train_labels)