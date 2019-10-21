# Simple Linear Regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset = pd.read_csv('Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Splitting dataset into Test and Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train, X_test, y_train, y_test)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train, X_test)
'''

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = lr.predict(X_test)

# Visualizing the Training Set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set) ')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set) ')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

lr.score(X_test, y_test)