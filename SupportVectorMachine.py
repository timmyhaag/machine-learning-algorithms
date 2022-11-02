# Created by Timothy Haag

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Read in data
digits = pd.read_csv("mnist_subset.csv")

# Setting features and label
X = digits.drop('7', axis=1)
y = digits['7']

# Split into training set and test set based on designated ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Train SVM models with a linear kernel and a rbf kernel
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf2 = svm.SVC(kernel='rbf')
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# Output accuracies
print("Accuracy for linear kernel:", metrics.accuracy_score(y_test, y_pred))
print("Accuracy for rbf kernel:", metrics.accuracy_score(y_test, y_pred2))
