import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,
r2_score
data=pd.read_csv("Employee_Details.csv")
print(data.head())
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","numbe
r_project",

"average_montly_hours","time_spend_company",
"Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,te
st_size=0.2,random_state=2)

lrr = LinearRegression()
lrr.fit(x_train, y_train)
y_lrr_pred = lrr.predict(x_test)

mse = mean_squared_error(y_test, y_lrr_pred)
r2 = r2_score(y_test, y_lrr_pred)

print("Linear Regression MSE:", mse)
print("Linear Regression R2:", r2)
