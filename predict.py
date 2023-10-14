import streamlit as st
import pandas as pd
import numpy as np

import vars
from predict_methods import *

from sklearn.model_selection import train_test_split


def app():
    # Splitting the data into training and testing sets.
    x = vars.df[vars.df.columns[:-1]]
    y = vars.df[vars.df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

    # select classifier
    st.subheader("Classification")
    classifier = st.selectbox("Select Classifier:", ("SVC", "Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"))
    
    vars.state = st.number_input("Random State:", value=1, step=1)

    # SUPPORT VECTOR MACHINE
    if classifier == "SVC":

        # parameters
        option = st.radio("Do you want to optimize model using hyperparameter tuning?", ("Yes", "No"))
        if option == "No":
            c = st.number_input("C Error Rate:", 1, 100, step=1)
            kernel = st.radio("Kernel:", ("linear", "rbf", "poly"))
            gamma = st.number_input("Gamma:", 1, 100, step=1)
        else:
            c, gamma, kernel = hyperparameter_tuning(X_train, y_train, "SVC")

        if st.button("Classify"):
            svc, svc_score, svc_predict = support_vector_machine(X_train, y_train, X_test, c, kernel, gamma)
            vars.info = []
            vars.info.extend([svc, svc_score, svc_predict])

    # LOGISTIC REGRESSION
    elif classifier == "Logistic Regression":
        lr, lr_score, lr_predict = logistic_regression(X_train, y_train, X_test)
        vars.info = []
        vars.info.extend([lr, lr_score, lr_predict])

    # RANDOM FOREST CLASSIFIER
    elif classifier == "Random Forest Classifier":

        # parameters
        option = st.radio("Do you want to optimize model using hyperparameter tuning?", ("Yes", "No"))
        if option == "No":
            n_estimators = st.number_input("Number of trees in the forest:", 1, 100, step=1)
            max_depth = st.number_input("Maximum depth of the tree:", 1, 100, step=1)
        else:
            n_estimators, max_depth = hyperparameter_tuning(X_train, y_train, "Random Forest Classifier")

        if st.button("Classify"):
            rfc, rfc_score, rfc_predict = random_forest_classifier(X_train, y_train, X_test, n_estimators, max_depth)
            vars.info = []
            vars.info.extend([rfc, rfc_score, rfc_predict])

    # DECISION TREE CLASSIFIER
    elif classifier == "Decision Tree Classifier":
        if len(vars.df.iloc[:,-1].value_counts()) == 2:
            # parameters
            option = st.radio("Do you want to optimize model using hyperparameter tuning?", ("Yes", "No"))
            if option == "No":
                criterion = st.selectbox("Criterion:", ("gini", "entropy"))
                max_depth = st.number_input("Maximum depth of the tree:", 1, 20, step=1)
            else:
                criterion, max_depth = hyperparameter_tuning(X_train, y_train, "Decision Tree Classifier")

            if st.button("Classify"):
                dtree, dtree_score, dtree_predict = decision_tree_classifier(X_train, y_train, X_test, y_test, criterion, max_depth)
                vars.info = []
                vars.info.extend([dtree, dtree_score, dtree_predict])
                vars.dtree_bool = True
        else:
            st.write("Cannot classify this data using decision tree since there are more than 2 labels.")
            
    # POLYNOMIAL DATA CLASSIFIER
    elif classifier == "Polynomial Data Classifier":
        option = st.radio("Do you want to optimize model using cross validation?", ("Yes", "No"))
        if option == "No":
            degree = st.number_input("Degree:", 1, 20, step=1)
        else:
            folds = st.number_input("How many folds in cross validation?", 5, 1)
            if st.button("Cross validate"):
                degree = cross_validiation(folds)

        if st.button("Classify"):
            poly, poly_score, poly_pred = polynomial_classifier(X_train, y_train, X_test, y_test, degree)
            vars.info = []
            vars.info.extend([poly, poly_score, poly_pred])


        


    # if classified, allow user to predict and plot confusion matrix
    if vars.classified == True:
        if st.checkbox("Plot Confusion Matrix"):
            confusion(vars.info[0], X_test, vars.info[2])


        # get values for prediction
        st.subheader("Select values:")
        values = []
        for i, j in zip(vars.df.columns[:-1], vars.df.dtypes[:-1]):
            if j == int:
                val = st.slider(str(i), int(vars.df[i].min()), int(vars.df[i].max()))
            elif j == float:
                val = st.slider(str(i), float(vars.df[i].min()), float(vars.df[i].max()))
            values.append(val)

        
        if st.button("Predict"):
            st.subheader("Prediction:")
            pred, test_r2_score, test_mae, test_msle, test_rmse = prediction(vars.info[0], vars.info[2], y_test, values)
            st.write("Predicted price:", pred)
            st.write("Accuracy:", vars.info[1])
            st.write("R2 score:", test_r2_score)
            st.write("Mean Absolute Erorr:", test_mae)
            st.write("Mean Squared Log Error:", test_msle)
            st.write("Root Mean Squared Erorr:", test_rmse)