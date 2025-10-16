from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,
cross_val_score
X, y = datasets.load_iris(return_X_y=True)
clf1 = DecisionTreeClassifier(random_state=42)
clf2 = LogisticRegression(max_iter=1000)
k_folds = KFold(n_splits = 10)
scores1 = cross_val_score(clf1, X, y, cv = k_folds)
scores2 = cross_val_score(clf2, X, y, cv = k_folds)
print("Cross Validation Scores: ", scores1)
print("Cross Validation Scores: ", scores2)
print("Average CV Score: ", scores1.mean())
print("Average CV Score: ", scores2.mean())
print("Number of CV Scores used in Average: ",
len(scores1))
print("Number of CV Scores used in Average: ",
len(scores2))
