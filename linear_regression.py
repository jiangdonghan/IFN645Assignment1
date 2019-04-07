#Task 3. Predictive Modeling Using Regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])


y = kick['IsBadBuy']
X = kick.drop(['IsBadBuy'], axis=1)
rs = 10
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, stratify=y, random_state=rs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)
model_default = LogisticRegression(random_state=rs)
model_default.fit(X_train, y_train)
print("Train accuracy:", model_default.score(X_train, y_train))
print("Test accuracy:", model_default.score(X_test, y_test))
y_pred = model_default.predict(X_test)
print(classification_report(y_test, y_pred))
params = {'C': [pow(10, x) for x in range(-6, 4)]}
model_cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
model_cv.fit(X_train, y_train)
print("Train accuracy:", model_cv.score(X_train, y_train))
print("Test accuracy:", model_cv.score(X_test, y_test))
y_pred = model_cv.predict(X_test)
print(classification_report(y_test, y_pred))
print(model_cv.best_params_)
#3.3k
coef = model_default.coef_[0]
feature_names = X.columns
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)
indices = indices[:119]
for i in indices:
    print(feature_names[i], ':', coef[i])
print('--------------------------')
coef = model_cv.best_estimator_.coef_[0]
feature_names = X.columns
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)
indices = indices[:119]
for i in indices:
    print(feature_names[i], ':', coef[i])
#3.4
print('Recursive Feature Elimination')
rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
rfe.fit(X_train, y_train)
print(rfe.n_features_)
X_train_sel = rfe.transform(X_train)
X_test_sel = rfe.transform(X_test)
params = {'C': [pow(10, x) for x in range(-6, 4)]}
cv_rfe = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv_rfe.fit(X_train_sel, y_train)
print("Train accuracy:", cv_rfe.score(X_train_sel, y_train))
print("Test accuracy:", cv_rfe.score(X_test_sel, y_test))
y_pred = cv_rfe.predict(X_test_sel)
print(classification_report(y_test, y_pred))
print(cv_rfe.best_params_)
coef = cv_rfe.best_estimator_.coef_[0]
feature_names = X.columns
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)
indices = indices[:46]
for i in indices:
    print(feature_names[i], ':', coef[i])
print('Selection by model')
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 7),
          'min_samples_leaf': range(10, 200, 10)}
dt_sbm = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10, n_jobs=-1)
dt_sbm.fit(X_train, y_train)
analyse_feature_importance(dt_sbm.best_estimator_, X.columns)
selectmodel = SelectFromModel(dt_sbm.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train)
X_test_sel_model = selectmodel.transform(X_test)
print(X_train_sel_model.shape)
params = {'C': [pow(10, x) for x in range(-6, 4)]}
cv_sbm = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv_sbm.fit(X_train_sel_model, y_train)
print("Train accuracy:", cv_sbm.score(X_train_sel_model, y_train))
print("Test accuracy:", cv_sbm.score(X_test_sel_model, y_test))
y_pred = cv_sbm.predict(X_test_sel_model)
print(classification_report(y_test, y_pred))
print(cv_sbm.best_params_)
coef = cv_sbm.best_estimator_.coef_[0]
feature_names = X.columns
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)
indices = indices[:46]
for i in indices:
    print(feature_names[i], ':', coef[i])