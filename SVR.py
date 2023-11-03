import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\sayya\Desktop\classs\16th,17th\EMP SAL.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(test_size=0.20,random_state=0)

from sklearn.svm import SVR
regressor=SVR(kernel='poly',degree=5,gamma='scale')
regressor.fit(x,y)

'''
y_pred=regressor.predict([[6.5]])

from sklearn.svm import SVR
regressor=SVR(kernel='linear',degree=5,gamma='scale')
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])
'''
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

import statsmodels.api as sm





