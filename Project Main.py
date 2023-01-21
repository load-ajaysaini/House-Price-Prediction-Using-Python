import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


housing = pd.read_csv("house_pricing.csv")
print(housing.head())
print(end="\n")

print("******************************")

print(housing.info())
print(end="\n")  

print("*********************************")

print("Total number of rows and columns in the dataset are : " , housing.shape , "respectively")
print(end="\n")

print("**********************************")

print("Stastical data is as follows : ")
print(housing.describe())
print(end="\n")

print("*********************************")


housing.dropna(inplace=True)
housing.dropna(how="all" , inplace=True)
print("Revised number of rows and columns after dropping null values " , housing.shape , "respectively")
print(end="\n")

print("*********************************")

print(housing.isnull().sum()) 
print(end="\n")

print("***********************************")

real_x = housing.iloc[: , 0:6].values  
real_y = housing.iloc[: , 6].values

x_train , x_test , y_train , y_test = train_test_split(real_x ,  real_y , test_size=0.2 , random_state=0)


le = LinearRegression()
le.fit(x_train , y_train)

print(end="\n")
print("Actual Price list of Houses : " , y_test)
print(end="\n")

print("*********************************")

pred_y = le.predict(x_test)
print(end="\n")
print("Predicted Price list of houses : " , pred_y)
print(end="\n")

print("*********************************")

accuracy=le.score(x_test , y_test)
print(accuracy*100,'%')
print(end="\n")


plt.scatter(y_test[0:50] , pred_y[0:50] , color='blue')
plt.title("Regression Graph for the given model ")
plt.xlabel("Actual values ")
plt.ylabel("Predicted values ")
plt.show()