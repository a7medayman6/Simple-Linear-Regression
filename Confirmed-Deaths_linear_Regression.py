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
#print("X_TEST")
#print(x_test)
#print("Y_TEST")
#print(y_test)
regr = LinearRegression()
regr.fit(x_train, y_train)
print("\tScore")
print(regr.score(x_test, y_test))

y_pred = regr.predict(x_test)
#print(y_pred)
plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
plt.show()
#val = np.array((1, 18)).reshape(-1, 1)
#print(val)
#print(regr.predict(val))
