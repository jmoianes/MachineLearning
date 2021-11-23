#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:29:04 2016

@author: softdevelopmentbcn
dataset: sonar dataset
"""

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data', header=None)
print(df.tail())

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[:, [60]].values
y
y.shape
y = np.where(y == 'R', -1, 1)
y
X = df.iloc[:, 0:60].values
X
X.shape

from perceptron import Perceptron

ppn = Perceptron(eta=0.001, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
