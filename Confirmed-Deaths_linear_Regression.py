import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('covid-19_Turkey.csv')
print(df.head())

df_binary = df[['Confirmed', 'Deaths']]
print("\n")
print(df_binary.head())
confirmed = df['Confirmed']
deaths = df['Deaths']

sns.lmplot(x = "Confirmed", y = "Deaths", data = df_binary, order = 2, ci = None)
plt.show()

confirmed.fillna(method = 'ffill', inplace = True)
deaths.fillna(method = 'ffill', inplace = True)
confirmed.dropna(inplace = True)
deaths.dropna(inplace = True)

x = np.array(confirmed).reshape(-1, 1)
y = np.array(deaths).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

regr = LinearRegression()
regr.fit(x_train, y_train)
print(regr.score(x_train, y_train))

y_pred = regr.predict(x_test)
plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
val = np.array(1529).reshape(-1, 1)
print(regr.predict(val))
plt.show()



#,Turkey,1529,37,0,3/23/2020