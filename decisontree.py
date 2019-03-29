# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:21:08 2018

@author: Manit
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')

X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

y_pred=regressor.predict(7)

plt.plot(X,Y)
plt.show()

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid, regressor.predict(X_grid))
plt.show()