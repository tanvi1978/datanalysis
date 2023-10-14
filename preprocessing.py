import streamlit as st
import pandas as pd 
import numpy as np
from preprocessing_methods import *

import vars

def app():
    st.title("Preprocessing")
    st.write("Select the ways with which you would like to preprocess the dataset.")

    choices = st.multiselect("Select issues with data:", ("NaN Values", "Categorical Data", "Drop Columns", "Rename Columns", "Normalize Features", "Oversampling", "Undersampling" ))

    if "Categorical Data" in choices:
        if len(vars.df.select_dtypes(exclude=np.number).columns) == 0:
            st.write("No categorical values.")
        else:
            vars.df = categorical(vars.df)

    if "NaN Values" in choices: 
        if len(vars.df.select_dtypes(exclude=np.number).columns) > 0:
            st.write("Please replace the categorical values first.")
        elif vars.df.isnull().sum().sum() == 0:
            st.write("No NaN values in DataFrame")
        else:
            vars.df = nan(vars.df)   
        
    if "Rename Columns" in choices:
        vars.df = renamer(vars.df)

    if "Drop Columns" in choices:
        vars.df = dropper(vars.df)

    if "Normalize Features" in choices:
        vars.df = normalize_features(vars.df)
    
    if "Oversampling" in choices:
        vars.df = oversample(vars.df)

    if "Undersampling" in choices:
        vars.df = undersample(vars.df)



