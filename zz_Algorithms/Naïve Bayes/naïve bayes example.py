#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 07:22:41 2016

@author: softdevelopmentbcn
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

dataset = pd.read_csv('../../data/02_Classification/01_weather/weather.csv')

# Calculate the class probabilities
p_class_go_out = dataset[dataset['class']=='go-out']['class'].count() / dataset['class'].count()
p_class_stay_home = dataset[dataset['class']=='stay-home']['class'].count() / dataset['class'].count()

# Calculate the conditional probabilities
p_weather_sunny_class_go_out =  dataset[(dataset['weather']=='sunny') & (dataset['class']=='go-out')]['class'].count() / (dataset[dataset['class']=='go-out']['class'].count())
p_weather_rainy_class_go_out =  dataset[(dataset['weather']=='rainy') & (dataset['class']=='go-out')]['class'].count() / (dataset[dataset['class']=='go-out']['class'].count())
p_weather_sunny_class_stay_home =  dataset[(dataset['weather']=='sunny') & (dataset['class']=='stay-home')]['class'].count() / (dataset[dataset['class']=='stay-home']['class'].count())
p_weather_rainy_class_stay_home =  dataset[(dataset['weather']=='rainy') & (dataset['class']=='stay-home')]['class'].count() / (dataset[dataset['class']=='stay-home']['class'].count())
p_car_working_class_go_out =  dataset[(dataset['car']=='working') & (dataset['class']=='go-out')]['class'].count() / (dataset[dataset['class']=='go-out']['class'].count())
p_car_broken_class_go_out =  dataset[(dataset['car']=='broken') & (dataset['class']=='go-out')]['class'].count() / (dataset[dataset['class']=='go-out']['class'].count())
p_car_working_class_stay_home =  dataset[(dataset['car']=='working') & (dataset['class']=='stay-home')]['class'].count() / (dataset[dataset['class']=='stay-home']['class'].count())
p_car_broken_class_stay_home =  dataset[(dataset['car']=='broken') & (dataset['class']=='stay-home')]['class'].count() / (dataset[dataset['class']=='stay-home']['class'].count())

# Make predictions




# Using sklearn
# Transform inputs
X  = dataset.copy()
y = X['class'].astype('category').cat.codes
X.drop(['class'], axis=1, inplace=True)
X['weather'] = X['weather'].astype('category').cat.codes
X['car'] = X['car'].astype('category').cat.codes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, y)

print 'score: {0}'.format(gnb.score(X, y))

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X, y)

print 'score: {0}'.format(bnb.score(X, y))