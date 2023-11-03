import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\sayya\Desktop\classs\JUNE1st\5. DECESSION TREE CODE\Social_Network_Ads.csv')
x=dataset.iloc[:,2:4].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import Normalizer
sc=Normalizer()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.naive_bayes import 
gb=GaussianNB()
gb.fit(x_train,y_train)

gb_y_pred=gb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,gb_y_pred)
print(cm1)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(y_test,gb_y_pred)
print(ac1)

from sklearn.naive_bayes import BernoulliNB()
bn=BernoulliNB()
bn.fit(x_train,y_train)

y_pred2=bn.predict(x_test)

ac2=accuracy_score(y_test, y_pred2)
print(ac2)








