#Task 4. Predictive Modelling Using Neural Networks
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
rs = 10
y = kick['IsBadBuy']
X = kick.drop(['IsBadBuy'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, stratify=y, random_state=rs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)
model = MLPClassifier(random_state=rs)
model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(model)
print(X_train.shape)
params = {'hidden_layer_sizes': [(x,) for x in range(5, 120, 20)]}
cv_nn = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_nn.fit(X_train, y_train)
print("Train accuracy:", cv_nn.score(X_train, y_train))
print("Test accuracy:", cv_nn.score(X_test, y_test))
print(cv_nn.best_params_)
params = {'hidden_layer_sizes': [(3,), (5,), (7,), (9,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}
cv_nn = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_nn.fit(X_train, y_train)
print("Train accuracy:", cv_nn.score(X_train, y_train))
print("Test accuracy:", cv_nn.score(X_test, y_test))
y_pred = cv_nn.predict(X_test)
print(classification_report(y_test, y_pred))
print(cv_nn.best_params_)
print(cv_nn.best_estimator_)
print("Number of iterations it ran: ", cv_nn.best_estimator_.n_iter_)
#4.3
print('RFE')
cv_nn_rfe = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_nn_rfe.fit(X_train_sel, y_train)
print("Train accuracy:", cv_nn_rfe.score(X_train_sel, y_train))
print("Test accuracy:", cv_nn_rfe.score(X_test_sel, y_test))
print(cv_nn_rfe.best_params_)
print(cv_nn_rfe.best_estimator_)
print('Selection by model')
cv_nn_sbm = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_nn_sbm.fit(X_train_sel_model, y_train)
print("Train accuracy:", cv_nn_sbm.score(X_train_sel_model, y_train))
print("Test accuracy:", cv_nn_sbm.score(X_test_sel_model, y_test))
print(cv_nn_sbm.best_params_)
print(cv_nn_sbm.best_estimator_)
analyse_feature_importance(dt_sbm.best_estimator_, X.columns)
print("Number of iterations it ran: ", cv_nn_sbm.best_estimator_.n_iter_)