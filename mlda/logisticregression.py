import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
y_lr_pred=lr.predict(x_test)
print("y_test shape:", y_test.shape)
print("y_lr_pred shape:", y_lr_pred.shape)
accuracy_lr=accuracy_score(y_test,y_lr_pred)
print(accuracy_lr)
