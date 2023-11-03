import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\sayya\Desktop\classs\16th,17th\1.POLYNOMIAL REGRESSION\emp_sal.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures()
p=pol_reg.fit_transform(x)
pol_reg.fit(p,y)

lin_reg2=LinearRegression()
lin_reg2.fit(p,y)


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('truth of bluff')
plt.xlabel('level of Emp')
plt.ylabel('Salary of Emp ')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(pol_reg.fit_transform(x)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('level of Emp')
plt.ylabel('Salary of Emp ')
plt.show()


