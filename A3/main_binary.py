import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from utilities import read_data, class_name_to_label, get_metrics

parser = argparse.ArgumentParser(
                    prog = 'Decision Trees and Random Forests',
                    description = 'COL341 Assignment 3',
                    epilog = 'by Chinmay Mittal')

parser.add_argument("--train_path", action="store", type=str, required=True, dest="train_path", help="path to train folder")
parser.add_argument("--val_path", action="store", type=str, required=False, dest="val_path", default=None, help="path to val folder")
parser.add_argument("--test_path", action="store", type=str, required=True, dest="test_path", help="path to test folder")
parser.add_argument("--out_path", action="store", type=str, required=True, dest="out_path", help="path to store the csv file")
parser.add_argument("--section", action="store", type=str, required=False, default="B", dest="section", help="...")

args = parser.parse_args()

X_train, y_train = read_data(args.train_path)
X_val, y_val = read_data(args.val_path)

#### set to face classification
y_train = (y_train == class_name_to_label["person"]).astype(np.int32)
y_val = (y_val == class_name_to_label["person"]).astype(np.int32)
####

print(f"Training Data: {X_train.shape}")
print(f"Valdiation Data: {X_val.shape}")

#### params
visualization = False
grid_search = False
trained = False
tree_params = {
    "criterion" : "gini",
    "max_depth" : 10,
    "min_samples_split" : 7,
}
####

if args.section == "B":
    ### sk-learn decision trees
    model = DecisionTreeClassifier(**tree_params)
    

elif args.section == "C":

    ### feature selection
    clf = DecisionTreeClassifier()
    feature_selector = SelectFromModel(clf, max_features=10)
    feature_selector.fit(X_train, y_train)
    selected_features = feature_selector.get_support()
    X_train = X_train[:, selected_features]
    X_val = X_val[:, selected_features]
    model = DecisionTreeClassifier()
    print(f"Number of Features Selected: {np.sum(selected_features)}")
    visualization = True
    grid_search = True

#### GRID SEARCH
if grid_search:
    parameters = dict(criterion=["gini", "entropy"],
                    max_depth=[None, 5, 7, 10, 15],
                    min_samples_split=[2,4,7,9])
    GS = GridSearchCV(model, parameters, cv=5)
    GS.fit(X_train, y_train)
    print('Best Criterion:', GS.best_estimator_.get_params()['criterion'])
    print('Best Max Depth:', GS.best_estimator_.get_params()['max_depth'])
    print('Best Min Samples Split:', GS.best_estimator_.get_params()['min_samples_split'])
    model = GS.best_estimator_
    trained = True

### TRAINING
if not trained:
    training_start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_end_time = time.perf_counter()
    training_time_taken_ms = (training_end_time - training_start_time) * 1000
    print(f"Training Time Section: {args.section} Time: {training_time_taken_ms:.4f} ms")

### INFERENCE
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
train_acc, train_precision, train_recall = get_metrics(y_true=y_train, y_pred=train_pred)
val_acc, val_precision, val_recall = get_metrics(y_true=y_val, y_pred=val_pred)

### METRICS
print()
print(f"Training Metrics   | Accuracy: {train_acc:.4f}% | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
print(f"Validation Metrics | Accuracy: {val_acc:.4f}% | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")



#### TREE VISUALIZATION
if visualization:
    plt.figure(figsize=(12,8))
    _ = tree.plot_tree(model, filled=True)
    plt.show()
    

