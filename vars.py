import pandas as pd
import numpy as np
import streamlit as st


# data
df = pd.DataFrame()

# check if preprocessed 
pre = False

# file name for downloader
file_name = ""

# check if classified 
classified = False

# contains model, accuracy score, and predicted values
info = [] 

# random state
state = None

# check if decision tree
dtree_bool = False


params_dict = {
    "SVC": {"C":np.arange(1, 10), "gamma":np.arange(1,10), "kernel":["linear", "rbf", "poly"]},
    "Random Forest Classifier": {"n_estimators":np.arange(1,10), "max_depth":np.arange(1,10)},
    "Decision Tree Classifier": {'criterion':['gini','entropy'], 'max_depth':np.arange(4,21), 'random_state': [42]},
    "Polynomial Data Classifier": {"a":"a"}
}


# df loader
def load_df(file):
    global df
    df = pd.read_csv(file)


