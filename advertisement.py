#problem statement:
''' finding which department shows much impacting on sales'''
import pandas as pd
data=pd.read_csv("E:\Advertising.csv")
print(data.info())
#print(data.isnull())
#print(data.describe())
import matplotlib.pyplot as plt
plt.scatter(data['TV'],data['sales'])
#plt.show()
plt.scatter(data['radio'],data['sales'])
#plt.show()
plt.scatter(data['newspaper'],data['sales'])
#plt.show()
x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
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
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="blue")
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
print("summery of the dataset tv versus sales:\n",final_out)




import pandas as pd
data=pd.read_csv("E:\Advertising.csv")
#print(data.info())
#print(data.isnull())
#print(data.describe())
import matplotlib.pyplot as plt
x=data.iloc[:,2:3].values
y=data.iloc[:,-1].values
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
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,y_pred,color="green")
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
print("summery of the dataset:\n",final_out)




import pandas as pd
data=pd.read_csv("E:\Advertising.csv")
#print(data.info())
#print(data.isnull())
#print(data.describe())
import matplotlib.pyplot as plt
x=data.iloc[:,3:4].values
y=data.iloc[:,-1].values
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
plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,y_pred,color="red")
plt.show()
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
print("summery of the dataset:\n",final_out)

