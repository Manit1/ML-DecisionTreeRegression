# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:28:25 2018

@author: Aastha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')

x=data.iloc[:,1:2].values
y=data.iloc[:,2].values


#fitting decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting the value
y_pred=regressor.predict(8.6)


x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')