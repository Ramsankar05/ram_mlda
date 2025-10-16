import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_dt_pred=dt.predict(x_test)
print("y_test shape:", y_test.shape)
print("y_dt_pred shape:", y_dt_pred.shape)
accuracy_dt=accuracy_score(y_test,y_dt_pred)
print(accuracy_dt)
