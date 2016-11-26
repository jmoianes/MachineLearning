#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:29:04 2016

@author: Julio Moyano
"""

'''
Data Set : Breast Cancer
Attribute Information:

1. mean radius
2. mean texture
3. mean perimeter
4. mean area
5. mean smoothness
6. mean compactness
7. mean concavity
8. mean concave points
9. mean symmetry
10. mean fractal dimension
11. radius error
12. texture error
13. perimeter error
14. area error
15. smoothness error
16. compactness error
17. concavity error
18. concave points error
19. symmetry error
20. fractal dimension error
21. worst radius
22. worst texture
23. worst perimeter
24. worst area
25. worst smoothness
26. worst compactness
27. worst concavity
28. worst concave points
29. worst symmetry
30. worst fractal dimension
31. class:
-- Malignant
-- Benign
'''
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = cancer['data']
y = cancer['target']

import pandas as pd

column_names = cancer.feature_names
df = pd.DataFrame(X, columns=column_names)
df['class'] = pd.Series(y)
print('Dataset: Breast Cancer')
print('---------------------')
print('Some values (tail)...')
print(df.tail())
print('\nFeature Types')
print('---------------')
print(df.dtypes)
print('\nFeature Descriptions')
print('----------------------')
print(df.describe())
print('\nTarget Description')
print('--------------------')
print(df['class'].describe())
print(df['class'].value_counts())

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot') # look pretty

# some visualizations
#print('\nData Visualization: Parallel Coordinates')
#from pandas.tools.plotting import parallel_coordinates
#
#plt.figure()
#parallel_coordinates(df, 'class')
#plt.show()

# target
y
# features
X

# train test dataset splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)   # test_size = 0.25 by default

# data preprocessing
# The neural network may have difficulty converging before the maximum number of iterations allowed if the data is not normalized. 
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
# Note that you must apply the same scaling to the test set for meaningful results. 
# There are a lot of different methods for normalization of data, we will use the built-in StandardScaler for standardization.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model: Multilayer Perceptron from sklearn package
print('\nUsing a Multilayer Perceptron from Sklearn')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), solver='lbfgs') # we choose 3 layers with the same number of neurons as there are features in our data set
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

# metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# using our perceptron built from scratch
from perceptron import Perceptron
print('\nUsing our Perceptron')
ppn = Perceptron(eta=0.01, n_iter=200)
y_train = np.where(y == 0, -1, 1)
y_test = np.where(y == 0, -1, 1)
ppn.fit(X_train, y_train)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.title('error evaluation', fontsize=12)
plt.show()

print('\nModel coefficients {0}'.format(ppn.w_))
print('Model accuracy {0}'.format(ppn.score(X_test, y_test)))

