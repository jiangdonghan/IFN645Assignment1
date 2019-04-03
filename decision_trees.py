# 2 Predictive Modeling Using Decision Trees
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import _tree
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

print('2.1')
print('Creating Decision Tree Model')
y = df['IsBadBuy']
X = df.drop(['IsBadBuy'], axis=1)
rs = 10
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

model = DecisionTreeClassifier(random_state=rs)
model.fit(X_train, y_train)

print('2.1a')
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print('Classification Report:')
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print('2.1b')
print('Total nodes(internal nodes + leaves: ', model.tree_.node_count)
print('Nodes: ', len([x for x in model.tree_.feature if x != _tree.TREE_UNDEFINED]))

print('2.1c')
print('Leaves: ', len([x for x in model.tree_.feature if x == _tree.TREE_UNDEFINED]))

##2.1d
##two methods we explored
## first, by exporting the visual representation of the first 5 levels (max_depth)
## (the whole tree is too large and incomprehensible to explore visually)
print('2.1d')
dotfile = StringIO()
export_graphviz(model, out_file=dotfile, max_depth=5, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
print('Outputting casestudy1.png')
graph.write_png("casestudy1.png")

## second, by coding a recursive print of the characteristics of the tree object ('tree_')
node_depth = np.zeros(shape=model.tree_.node_count, dtype=np.int64)
is_leaves = np.zeros(shape=model.tree_.node_count, dtype=bool)
stack = [(0, -1)]
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (model.tree_.children_left[node_id] != model.tree_.children_right[node_id]):
        stack.append((model.tree_.children_left[node_id], parent_depth + 1))
        stack.append((model.tree_.children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
print("The model has the following tree structure:")
for i in range(10):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test: go to node %s if %s[:, %s] <= %s else to node %s."
              % (node_depth[i] * "\t",
                 i,
                 model.tree_.children_left[i],
                 X.columns[model.tree_.feature[i]],
                 model.tree_.feature[i],
                 model.tree_.threshold[i],
                 model.tree_.children_right[i],
                 ))
print()

print('2.1e')
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)
indices = indices[:5]
for i in indices:
    print(feature_names[i], ':', importances[i])


print('2.1f')
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print('Answer: The numbers above suggest there is some overfitting. The model is perfectly aligned to the Training dataset and is less accurate on the Testing data.')

print('2.1g')
print('Creating plots comparing accuracy of hyperparameters')

# check the model performance for max depth from 2-20
print('1 of 2: max_depth...')
test_score = []
train_score = []
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))
plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()

# check the model performance for min samples split from 100-4000
print('2 of 2: min_samples_split...')
test_score = []
train_score = []
for min_samples_split in range(100, 4000, 50):
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))
plt.plot(range(100, 4000, 50), train_score, 'b', range(100, 4000, 50), test_score, 'r')
plt.xlabel('min_samples_split\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()

#max_depth = 3 seems to be the best performing tree
print('Creating new Decision Tree Model with optimal max_depth')
y = df['IsBadBuy']
X = df.drop(['IsBadBuy'], axis=1)
rs = 10
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

print('2.1ga')
model = DecisionTreeClassifier(max_depth=3, random_state=rs)
model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print('Classification Report:')
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print('2.1gb')
from sklearn.tree import _tree
print('Total (internal nodes + leaves): ', model.tree_.node_count)
print('Nodes: ', len([x for x in model.tree_.feature if x != _tree.TREE_UNDEFINED]))
print('2.1gc')
print('Leaves: ', len([x for x in model.tree_.feature if x == _tree.TREE_UNDEFINED]))
print('2.1gd')
print('two ways of showing this')
dotfile = StringIO()
export_graphviz(model, out_file=dotfile, max_depth=5, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
print('Outputting casestudy2.png')
graph.write_png("casestudy2.png")
print('or by coding a recursive print of the characteristics of the tree object (tree_)')
node_depth = np.zeros(shape=model.tree_.node_count, dtype=np.int64)
is_leaves = np.zeros(shape=model.tree_.node_count, dtype=bool)
stack = [(0, -1)]

while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (model.tree_.children_left[node_id] != model.tree_.children_right[node_id]):
        stack.append((model.tree_.children_left[node_id], parent_depth + 1))
        stack.append((model.tree_.children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
print("The model has the following tree structure:")
for i in range(10):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test: go to node %s if %s[:, %s] <= %s else to node %s."
              % (node_depth[i] * "\t",
                 i,
                 model.tree_.children_left[i],
                 X.columns[model.tree_.feature[i]],
                 model.tree_.feature[i],
                 model.tree_.threshold[i],
                 model.tree_.children_right[i],
                 ))
print()
print('2.1ge')
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)
indices = indices[:5]
for i in indices:
    print(feature_names[i], ':', importances[i])
print('2.1gf')
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print('Answer: No, this tree has almost identical results between Training and Test data.')

print('-----------------------------------------------------')
print('2.2')
from sklearn.model_selection import GridSearchCV
print('Building another decision tree tuned with GridSearchCV')
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 7),
          'min_samples_leaf': range(10, 100, 10)}
cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)
print('2.2a')
print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(cv.best_params_)
print('2.2b')
print('Total (internal nodes + leaves): ', cv.best_estimator_.tree_.node_count)
print('Nodes: ', len([x for x in cv.best_estimator_.tree_.feature if x != _tree.TREE_UNDEFINED]))
# 3 nodes. Less than previous models (6 nodes). I guess mostly due to the lower 'max_depth' (2 vs 3)
print('2.2c')
print('Leaves: ', len([x for x in cv.best_estimator_.tree_.feature if x == _tree.TREE_UNDEFINED]))
# 4 leaves
print('2.2d')
print('Outputting optimal_tree.png')
dotfile = StringIO()
export_graphviz(cv.best_estimator_, out_file=dotfile, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("optimal_tree.png")
print('Answer is same as default tree:"VehYear" is used for the first split. The competing splits are "VehBCost" and "WheelType_Alloy"')
print('2.2e')
importances = cv.best_estimator_.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)
indices = indices[:5]
for i in indices:
    print(feature_names[i], ':', importances[i])
print('2.2f')
print('Chances of model overfitting are very low. Test data/training data scores are almost even.')
print('2.2g')
#'criterion': ['gini', 'entropy'],
#'max_depth': range(2, 7),
#'min_samples_leaf': range(10, 100, 10)}
print('2.3')
print('2.4')
print('Answer: With some accuracy, we could describe potential ''kicks'' as:')
print('- cars older than 2006 that cost more than $4017, OR')
print('- cars newer than 2005 that don''t come with alloy wheels.')