import pandas as pd
import numpy as np
#from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("Employee_Details.csv")
print(data.head())
X = data[['average_montly_hours']] # Number of
working Hours
y = data['last_evaluation'] # Evaluation of Worker
# Split the data into training and testing sets (80%
train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Slope(m):", model.coef_)
print("Intercept(c):", model.intercept_)
# Plot the actual data points
plt.scatter(X_test, y_test, color='blue',
label='Actual')
# Plot the regression line
plt.plot(X_test, y_pred, color='red',
label='Regression Line')
# Add labels and title
plt.xlabel('Average Working Hours in a month')
plt.ylabel('Evaluation of Worker')
plt.title('Simple Linear Regression: Average Working
Hours in a month Vs Evaluation of Worker')
plt.legend()
plt.show()
