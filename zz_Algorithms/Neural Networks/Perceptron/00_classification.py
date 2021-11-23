#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:29:04 2016

@author: Julio Moyano
"""

'''
Data Set: Iris
Attribute Information:

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica
'''

import pandas as pd

column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=column_names)
print('Dataset: Iris Dataset')
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
print('\nData Visualization: Parallel Coordinates')
from pandas.tools.plotting import parallel_coordinates

plt.figure()
parallel_coordinates(df, 'class')
plt.show()

# target
print('we will use only two classes: iris-setosa and iris-versicolor')
y = df.iloc[0:100, 4].values
y
y.shape
y = np.where(y == 'Iris-setosa', -1, 1)
y
# features
print('we will use only two features for classification: sepal length and petal length')
X = df.iloc[0:100, [0,2]].values
X


plt.scatter(X[:50, 0], X[:50, 1], color='red',marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',marker='x', label='versicolor')
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.title('petal lenght vs sepal length', fontsize=12)
plt.show()

from perceptron import Perceptron

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.title('error evaluation', fontsize=12)
plt.show()

print('\nModel coefficients {0}'.format(ppn.w_))
print('Model accuracy {0}'.format(ppn.score(X, y)))

# Let's generalize the idea using the whole set of features (keeping a two class classification problem)
# features
print('we will use the whole set of features for classification: sepal length, sepal width, petal length and petal width')
X = df.iloc[0:100,0:4].values

ppn2 = Perceptron(eta=0.1, n_iter=10)
ppn2.fit(X, y)

plt.plot(range(1, len(ppn2.errors_)+1), ppn2.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.title('error evaluation', fontsize=12)
plt.show()

print('\nModel coefficients {0}'.format(ppn2.w_))
print('Model accuracy {0}'.format(ppn2.score(X, y)))

# let's use PCA for visualizing the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
Z = pca.transform(X)

plt.scatter(Z[:50, 0], Z[:50, 1], color='red',marker='o', label='setosa')
plt.scatter(Z[50:100, 0], Z[50:100, 1], color='blue',marker='x', label='versicolor')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend(loc='upper left')
plt.title('data visualization using PCA', fontsize=12)
plt.show()

# lets change the flowers class
# target
print('we will use only two classes: iris-versicolor and iris-virginica')
y = df.iloc[50:, 4].values
y
y.shape
y = np.where(y == 'Iris-versicolor', -1, 1)
y

#features
print('we will use only two features for classification: sepal length and petal length')
X = df.iloc[50:, [0,2]].values
X

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',marker='x', label='versicolor')
plt.scatter(X[:50, 0], X[:50, 1], color='green',marker='.', label='virginica')
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.title('petal lenght vs sepal length', fontsize=12)
plt.show()

print('as the previous chart shows, this is going to be more difficult to classify')

ppn3 = Perceptron(eta=0.1, n_iter=10)
ppn3.fit(X, y)

plt.plot(range(1, len(ppn3.errors_)+1), ppn3.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.title('error evaluation', fontsize=12)
plt.show()

print('\nModel coefficients {0}'.format(ppn3.w_))
print('Model accuracy {0}'.format(ppn3.score(X, y)))

print('\nUsing a higher number of iterations')
ppn3 = Perceptron(eta=0.01, n_iter=100)
ppn3.fit(X, y)

plt.plot(range(1, len(ppn3.errors_)+1), ppn3.errors_, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.title('error evaluation', fontsize=12)
plt.show()

print('\nModel coefficients {0}'.format(ppn3.w_))
print('Model accuracy {0}'.format(ppn3.score(X, y)))
