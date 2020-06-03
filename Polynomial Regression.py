# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:45:56 2020

@author: user
"""

#Importing libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Gaming_data.csv')
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1].values
dataset.describe()

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Fitting polinomial regression into dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4 )#Transforming tool
x_poly = poly_reg.fit_transform(x)
lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly,y)


#Visualising Linear Regression result
plt.scatter(x,y)
plt.plot(x, lin_reg.predict(x), color='red')
plt.title('Gaming Data(Linear Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')

#Visualising Polinomial Regression result
plt.scatter(x,y)
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color='red')
plt.title('Gaming Data(Polynomial Regression)')
plt.xlabel('Gaming Steps')
plt.ylabel('Points')

#Pedicting new result with linear regression
 lin_reg.predict([[7.5]])

#Pedicting new result with Polinomial regression
lin_reg2.predict(poly_reg.fit_transform([[7.5]]))
 

#predicting new result with polynimial regression