import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("E:\Admission_Prediction.csv")
x=data.iloc[:,1:8].values
y=data.iloc[:,-1].values
print(x.shape)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#imputer=imputer.fit(x[:,1:2])
x[:,:7]=imputer.fit_transform(x[:,:7])
#print(x)
from  sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
reg_obj=LinearRegression()
reg_obj.fit(x_train,y_train)
print("intercept value(c)\n",reg_obj.intercept_)
print("coefficient value(m)\n",reg_obj.coef_)
y_pred=reg_obj.predict(x_test)
output=pd.DataFrame({"actual":y_test,"predicted":y_pred})
print(output)
#plt.scatter(x_test,y_test,color="red")
#plt.plot(x_test,y_pred,color="blue")
#plt.show()
import numpy as np
from sklearn import metrics
mae=metrics.mean_absolute_error(y_test,y_pred)
mse=metrics.mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2e=metrics.r2_score(y_test,y_pred)
print('mean absolute error:\n',mae)
print('mean squared error:\n',mse)
print('root mean squared error:\n',rmse)
print('r squared error:\n',r2e)
import statsmodels.api as sm
from statsmodels.api import OLS
x=sm.add_constant(x)
final_out=OLS(y,x).fit().summary()
print("summery of the dataset  :\n",final_out)