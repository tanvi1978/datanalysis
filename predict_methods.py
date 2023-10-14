# general
import numpy as np
import pandas as pd
import streamlit as st

# plots
import matplotlib.pyplot as plt

# variables
import vars

# train test split
from sklearn.model_selection import train_test_split

# models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# hyperparamter tuning
from sklearn.model_selection import GridSearchCV  

from sklearn.model_selection import cross_val_score

# metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, plot_confusion_matrix, accuracy_score


# predict function
def prediction(model, y_test_predict, y_test, features:list):
    # predict value
    pred = model.predict([features])
    pred = pred[0]

    # calculate scores
    test_r2_score = r2_score(y_test, y_test_predict)
    test_mae = mean_absolute_error(y_test, y_test_predict)
    test_msle = mean_squared_log_error(y_test, y_test_predict)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))

    return pred, test_r2_score, test_mae, test_msle, test_rmse


def confusion(model, X_test, predict):
    plt.figure(figsize = (10, 6)) # create figure
    plot_confusion_matrix(model, X_test, predict, values_format = 'd') # plot
    st.pyplot() # show


def support_vector_machine(X_train, y_train, X_test, c, kernel, gamma):
    vars.classified=True
    st.subheader("SVC Model")
    svc = SVC(C=c, kernel=kernel, gamma=gamma, random_state=vars.state)
    svc.fit(X_train, y_train)
    svc_predict = svc.predict(X_test)
    svc_score = svc.score(X_test, svc_predict)

    return svc, svc_score, svc_predict


def logistic_regression(X_train, y_train, X_test):
    vars.classified=True
    st.subheader("Logistic Regression Model")
    lr = LogisticRegression(n_jobs=-1, random_state=vars.state)
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)
    lr_score = lr.score(X_test, lr_predict)
    
    return lr, lr_predict, lr_score


def random_forest_classifier(X_train, y_train, X_test, n_estimators, max_depth):
    rfc = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, random_state=vars.state)
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    rfc_score = rfc.score(X_test, rfc_predict)

    return rfc, rfc_score, rfc_predict


def decision_tree_classifier(X_train, y_train, X_test, criterion, max_depth):
    dtree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dtree.fit(X_train, y_train) 
    y_train_pred = dtree.predict(X_train)
    y_test_pred = dtree.predict(X_test)
    dtree_score = accuracy_score(y_train, y_train_pred)

    return dtree, dtree_score, y_test_pred


def polynomial_classifier(X_train, y_train, X_test, y_test, degree):
    pipe = [
            ('polynomial', PolynomialFeatures(degree = degree, include_bias = False)),
            ('linear', LinearRegression()) 
            ]
    
    pipeline_model = Pipeline(pipe)
    pipeline_model.fit(X_train.reshape(-1,1), y_train)
    pred = pipeline_model.predict(X_test.reshape(-1,1))
    score = r2_score(y_test, pred)

    return pipeline_model, pred, score


def hyperparameter_tuning(X_train, y_train, classifier_input: str):
    model_dict = {"SVC": SVC(), "Random Forest Classifier": RandomForestClassifier(), "Decision Tree Classifier": DecisionTreeClassifier() }
    grid = GridSearchCV(model_dict[classifier_input], vars.params_dict[classifier_input], scoring = 'roc_auc', n_jobs = -1)
    grid.fit(X_train, y_train)
    return grid.best_params_.values()

def cross_validiation(folds):
    best_score = 0
    best_degree = 0
    cv_list = []
    for degree in range(1, 31):
        # Apply polynomial regression for degrees 1 to 30
        pipe = [('polynomial', PolynomialFeatures(degree = degree, include_bias = False)),
                ('linear', LinearRegression())]

        # Create 'Pipeline' object. Apply 10 fold cross validation and store the accuracy scores for each fold.
        pipeline_model = Pipeline(pipe)
        scores = cross_val_score(pipeline_model, vars.df.iloc[:, :-1], vars.df.iloc[:, -1], cv=folds)
        # Append the mean accuracy score to the list.
        cv_list.append(scores.mean())
        # If maximum accuracy score > 'best_score', then store this score in 'best_score' and degree in 'best_degree'.
        if max(scores) > best_score:
            best_score = max(scores)
            best_degree = degree

    return best_degree

