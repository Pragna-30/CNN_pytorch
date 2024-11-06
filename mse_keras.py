from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 


california=fetch_california_housing()




X_train,X_test,Y_train,Y_test=train_test_split(california.data,california.target,test_size=0.2)


#normalize the data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#train the model
model=LinearRegression()
model.fit(X_train,Y_train)

output=model.predict(X_test)
mse=mean_squared_error(output,Y_test)

print(mse)

#mlp
