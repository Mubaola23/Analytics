# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:06:56 2020

@author: user
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Gaming_data.csv')
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values


#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#Fiting SVR to dataset
from sklearn.svm import SVR
regressor =SVR(kernel ='rbf')
regressor.fit(x,y)

#VIsualising SVR result

plt.scatter(x,y)
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('Gaming data(SVR)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')
plt.show()

#Predicting result

y_pred = regressor.predict(sc_x.transform([[9.5]]))
y_pred = sc_y.inverse_transform(y_pred)