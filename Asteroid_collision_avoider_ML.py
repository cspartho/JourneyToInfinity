# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:00:26 2019

@author: Linkon from Galactic Angels
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(r'C:\Users\Hp\Desktop\Nasa\Dataset\Asteroid_data_final.csv')

dataset.shape

dataset.describe()

dataset.isnull().any()
dataset = dataset.fillna(method='ffill')

X = dataset[['As_diam_km', 'As_dist_km', 'As_velocity_kmh', 'As_velocity_angle', 'As_dist_relative_flag','Coordinate_flag','Sc_diam_m', 'Sc_velocity_kmh', 'Relative_velocity', 'Estimated_time', 'New_theta']].values
y = dataset['Final_Angle'].values

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Final_Angle'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  s
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

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
