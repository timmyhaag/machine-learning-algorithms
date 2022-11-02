# Created by Timothy Haag

import pandas as pd
from sklearn import tree

# Loading data
train_data = pd.read_csv("Titanic_training.csv")
test_data = pd.read_csv("Titanic_test.csv")

PassengerId = test_data['PassengerId']

# Assigning an “unknown” value for missing values in data
test_data["Age"].fillna("unknown", inplace=True)
train_data["Age"].fillna("unknown", inplace=True)

test_data["Fare"].fillna("unknown", inplace=True)
train_data["Fare"].fillna("unknown", inplace=True)

test_data["Embarked"].fillna("unknown", inplace=True)
train_data["Embarked"].fillna("unknown", inplace=True)

# Outputs the head of each data set
print("The head of the train data are: \n", train_data.head())
print("The head of the test data are: \n", test_data.head())

# Combines data sets
combined_data = [train_data, test_data]

# Data is assigned numerical values where necessary
for dataset in combined_data:
    dataset['Sex'] = dataset['Sex'].replace({'female': 0, 'male': 1}).astype(int)
    dataset['Age'] = dataset['Age'].replace({'unknown': 0}).astype(int)
    dataset['Fare'] = dataset['Fare'].replace({'unknown': 0}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2, 'unknown': 0}).astype(int)

# Drops irrelevant information
drop_data = ['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin']
train_data = train_data.drop(drop_data, axis=1)
test_data = test_data.drop(drop_data, axis=1)

# Numpy arrays for designated model
y_train = train_data['Survived']
x_train = train_data.drop(['Survived'], axis=1).values
x_test = test_data.values

# Decision tree is implemented and prediction is made for the test dataset
decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)

# Output the total number of nodes and leaf nodes in the tree
tree_node_count = decision_tree.tree_
print("\nTotal number of leaf nodes: ", decision_tree.get_n_leaves())
print("Total number of nodes: ", tree_node_count.node_count)

# Designated results are saved to a csv file.
submission_data = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": y_pred
})
submission_data.to_csv('submission_data.csv', index=False)
