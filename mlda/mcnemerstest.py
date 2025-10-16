import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import
mcnemar
data=pd.read_csv("Employee_Details.csv")
#print(data.head())
# Features and target
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
#print(data.head())
x=data[["satisfaction_level","last_evaluation","numbe
r_project","average_montly_hours","time_spend_company
",
"Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
y.head()
# Train/test split
x_train,x_test,y_train,y_test=train_test_split(x,y,te
st_size=0.2,random_state=2)
# Decision Tree
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_dt_pred=dt.predict(x_test)
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
y_lr_pred=lr.predict(x_test)

print("Decision Tree Accuracy:",
accuracy_score(y_test, y_dt_pred))
print("Logistic Regression Accuracy:",
accuracy_score(y_test, y_lr_pred))

# ---------------------------
# McNemar’s Test
# ---------------------------
# Build contingency table
table = [[0,0],[0,0]]
for i in range(len(y_test)):
if y_test.iloc[i] == y_dt_pred[i] and
y_test.iloc[i] != y_lr_pred[i]:

table[0][1] += 1 # DT correct, LR wrong
elif y_test.iloc[i] != y_dt_pred[i] and
y_test.iloc[i] == y_lr_pred[i]:

table[1][0] += 1 # LR correct, DT wrong

for i in range(len(y_test)):
if y_test.iloc[i] == y_dt_pred[i] and
y_test.iloc[i] == y_lr_pred[i]:

table[0][0] += 1 # DT correct, LR correct
elif y_test.iloc[i] != y_dt_pred[i] and
y_test.iloc[i] != y_lr_pred[i]:

table[1][1] += 1 # LR wrong, DT wrong
print("Contingency Table (McNemar):", table)
# Run McNemar’s test
result = mcnemar(table, exact=True) # exact=True
recommended for small samples
print("McNemar’s Test Statistic:", result.statistic)
print("McNemar’s Test p-value:", result.pvalue)
if result.pvalue < 0.05:
print("Significant difference between models
(reject H0)")
else:
print("No significant difference between models
(fail to reject H0)")
