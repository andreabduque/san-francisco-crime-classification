import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("train.csv")

X = df.drop(axis=1, columns = ["Category"])
X = X.drop(axis=1, columns = ["DayOfWeek"])
y = df.Category

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)
print(skf)

clf = RandomForestClassifier(n_estimators = 100)

for train_index, test_index in skf.split(X, y):
	classifier = clf.fit(X.loc[train_index], y.loc[train_index])
	predictions = classifier.predict_proba(X.loc[test_index])
	l_l = log_loss(df.Category.loc[test_index], predictions)
	print(l_l)
	break

