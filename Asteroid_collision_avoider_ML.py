# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:00:26 2019

@author: Linkon
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(r'C:\Users\Hp\Desktop\Nasa\Final\New folder (2)\IRIS3.csv')

dataset.shape

dataset.describe()

dataset.isnull().any()
dataset = dataset.fillna(method='ffill')

X = dataset[['As_diam_m', 'As_dist_km', 'As_velocity_kmh', 'As_velocity_angle', 'As_dist_relative_flag', 'soron']].values
y = dataset['Rotate_angle'].values

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Rotate_angle'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
#coeff_df

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df.head(25)
df1


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()