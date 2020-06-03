#Data preprocessing

#Importing libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Org_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# working with missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Encoding Categories
#Encoding independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype =np.float)

#Avoid Dummy Variable
x =x[:,1:]

#spliting dataset into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.20, random_state=0)

#Fitting Multiple Linear Regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting test result
y_pred =regressor.predict(x_test)

#Optimizing model using backward elemination
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values =x,axis=1)

x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog =y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog =y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog =y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog =y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog =y, exog=x_opt).fit()
regressor_OLS.summary()