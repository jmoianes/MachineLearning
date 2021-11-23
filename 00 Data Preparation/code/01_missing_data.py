# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:21:50 2021

@author: JulioMoyanoGarcía
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('data/00_DataPreparation/missing_data_example.csv')

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

dataset[['Age', 'Salary']] = imputer.fit_transform(dataset[['Age', 'Salary']])
dataset
