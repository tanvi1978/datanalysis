import streamlit as st

def app():
    st.write(''' 
    
        It is recommened to first preprocess the data either using the built-in preprocessing feature
        or by yourself. Using categorical data will break some of the features.

        To preprocess, select the method and then follow the steps of that method.
        
        On the view data page, you can also download the preprocessed data.

        To visualize the data, select which graph you want to make and select the parameters.
        The decision tree will be avaiable if the decision tree classifier is made on the Predict page.

        On the predict page, you can either select your own parameters or use the built-in hyperparameter tuning which uses GridSearchCV.

    ''')