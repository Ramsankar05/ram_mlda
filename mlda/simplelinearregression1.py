import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = { 'x':[1,2,3,4,5],
'y':[2,4,5,4,5] }
df = pd.DataFrame(data)
X = df[['x']]
Y = df[['y']]
model = LinearRegression()
model.fit(X,Y)
y_pred = model.predict(X)
print("Slope(m):", model.coef_[0])
print("Intercept(c):", model.intercept_)
