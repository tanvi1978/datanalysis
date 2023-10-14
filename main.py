
import streamlit as st
import pandas as pd
import numpy as np


import home
import preprocessing
import data
import plots
import predict
import vars
import help




# page config
st.set_page_config(page_title = "Data Analysis Web App", 
                        page_icon = ":random:", 
                        layout = 'centered', 
                        initial_sidebar_state = 'auto')

# filter warning
st.set_option('deprecation.showPyplotGlobalUse', False)

state = st.empty()

file = state.file_uploader("Upload CSV file here:", type=["csv"])

if file is not None:
    if vars.df.empty:
        vars.file_name = file.name
        vars.load_df(file)
        state.empty()

    state.empty()

    # multi-page navigation system
    pages_dict = {"Home":home, "Preprocess":preprocessing, "View Data": data, "Visualize Data": plots, "Predict": predict, "Help":help}    
    page = st.sidebar.radio("Navigation", ("Home", "Preprocess", "View Data", "Visualize Data", "Predict", "Help"))
    pages_dict[page].app()
