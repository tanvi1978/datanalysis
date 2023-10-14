import pandas as pd
import numpy as np
import streamlit as st
import io

import vars

# This page allows users to view the data and other information/statistics about the data
def app():
    # dataframe
    st.title("View Data")
    st.dataframe(vars.df)

    # print df.info() 
    st.subheader("Data Info")
    buffer = io.StringIO()
    vars.df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # descriptive statistics
    st.subheader("Descriptive statistics:")
    if len(vars.df.select_dtypes(exclude=np.number)) == 0:
        st.write(vars.df.describe())
    else:
        st.write("Please replace categorical values to view descriptive statistics.")

    # download preprocessed data as csv
    st.subheader("Download data")
    st.write("Download the csv file of the preprocessed data.")
    csv = vars.df.to_csv(header=False, index=False, columns=list(vars.df.columns)).encode('utf-8')
    st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name= f"preprocessed_{vars.file_name}.csv",
     mime='csv',
    )




