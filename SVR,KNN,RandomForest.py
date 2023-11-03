import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\sayya\Desktop\classs\16th,17th\EMP SAL.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.neighbors import KNeighborsRegressor
knr=KNeighborsRegressor(n_neighbors=6,weights='distance')
knr.fit(x,y)

knr_pred1=knr.predict([[6.5]])
knr2=KNeighborsRegressor(n_neighbors=6,weights='distance',algorithm='ball_tree')
knr2.fit(x,y)
k2=knr2.predict([[6.5]])


from sklearn.svm import SVR
regressor=SVR(kernel='rbf',degree=2,gamma='scale')
regressor.fit(x,y)

svr_pred=regressor.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(x,y)

r_pred=rfr.predict([[6.5]])

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x,y)

dtr=dt.predict([[6.5]])















