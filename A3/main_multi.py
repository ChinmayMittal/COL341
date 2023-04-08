import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from utilities import read_data, get_metrics

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

print(f"Training Data: {X_train.shape}")
print(f"Valdiation Data: {X_val.shape}")

#### params
visualization = False
grid_search = False
trained = False
plot_confusion = False
average = "macro" ### for precision and recall
max_features = 10 ### for feature selection
verbose = 1
tree_params = {
    "criterion" : "gini",
    "max_depth" : 10,
    "min_samples_split" : 7,
}
####

if args.section == "A":
    ### sk-learn decision trees
    print("Sklearn Decision Trees")
    model = DecisionTreeClassifier(**tree_params)
elif args.section == "B":
    print("Feature Selection, Grid Search and Visualization ...")
    ### feature selection
    clf = DecisionTreeClassifier()
    feature_selector = SelectFromModel(clf, max_features=max_features)
    feature_selector.fit(X_train, y_train)
    selected_features = feature_selector.get_support()
    X_train = X_train[:, selected_features] ### reduced feature set
    X_val = X_val[:, selected_features]
    model = DecisionTreeClassifier()
    print(f"Number of Features Selected: {np.sum(selected_features)}")
    visualization = True
    grid_search = True

elif args.section == "C":
    print("Cost Complexity Pruning ... ")
    model = DecisionTreeClassifier(random_state=0)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post", color="red")
    ax.set_xlabel("Effective alpha")
    ax.set_ylabel("Total impurity of leaves")
    ax.set_title("Total Impurity vs Effective alpha for Training set")
    plt.grid(True)
    plt.show()
    plt.clf()
    clfs = []
    for i, ccp_alpha in enumerate(ccp_alphas):
        print(f"Training Pruned Trees {(i/len(ccp_alphas))*100:.2f}%", end="\r")
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    clfs, ccp_alphas = clfs[:-1], ccp_alphas[:-1] ### last tree is trivial tree with single node
    node_counts = [clf.tree_.node_count for clf in clfs]
    depths = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    val_scores = [clf.score(X_val, y_val) for clf in clfs]
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Number of nodes")
    ax[0].set_title("Number of nodes vs Alpha")
    ax[0].grid(True)
    ax[1].plot(ccp_alphas, depths, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Depth of tree")
    ax[1].set_title("Depth vs Alpha")
    ax[1].grid(True)
    fig.tight_layout()
    plt.show()
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs alpha for training and validation sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, val_scores, marker="o", label="val", drawstyle="steps-post")
    ax.grid(True)
    ax.legend()
    plt.show()
    max_val_index = np.array(val_scores).argmax()
    model = clfs[max_val_index]
    trained = True

elif args.section == "D":
    print("Random Forests")
    model = RandomForestClassifier()
    grid_search = True
### GRID SEARCH
if grid_search:
    print("Performing Grid Search ... ")
    if args.section == "B":
        ### decision trees
        parameters = dict(criterion=["gini", "entropy"],
                    max_depth=[None, 5, 7, 10, 15],
                    min_samples_split=[2,4,7,9])
    elif args.section == "D":
        ### random forests
        parameters = dict( n_estimators = [80,100,150,200],
                        criterion=["gini", "entropy"],
                        max_depth=[None, 5, 7, 10],
                        min_samples_split=[5,7,10])
    GS = GridSearchCV(model, parameters, cv=5, verbose=verbose)
    GS.fit(X_train, y_train)
    print('Best Criterion:', GS.best_estimator_.get_params()['criterion'])
    print('Best Max Depth:', GS.best_estimator_.get_params()['max_depth'])
    print('Best Min Samples Split:', GS.best_estimator_.get_params()['min_samples_split'])
    if(args.section == "D"):
        print('Best Number of Estimators:', GS.best_estimator_.get_params()['n_estimators'])
    model = GS.best_estimator_  ### best learnt parameters
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
train_acc, train_precision, train_recall = get_metrics(y_true=y_train, y_pred=train_pred, average=average)
val_acc, val_precision, val_recall = get_metrics(y_true=y_val, y_pred=val_pred, average=average)

## METRICS
print()
print(f"Training Metrics   | Accuracy: {train_acc:.4f}% | {average} Precision: {train_precision:.4f} | {average} Recall: {train_recall:.4f}")
print(f"Validation Metrics | Accuracy: {val_acc:.4f}% | {average} Precision: {val_precision:.4f} | {average} Recall: {val_recall:.4f}")

### TREE VISUALIZATION
if visualization:
    plt.figure(figsize=(12,8))
    _ = tree.plot_tree(model, filled=True)
    plt.show()