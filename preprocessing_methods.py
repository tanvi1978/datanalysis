
import streamlit as st
import numpy as np
import pandas as pd 
import vars

# normalize
from sklearn.preprocessing import StandardScaler

# sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def nan(df):
    way = st.selectbox("Select method to deal with missing values:", ("-----", "Delete rows", "Replace with mean", "Replace with median", "Replace with mode"))
    if way == "Delete rows": df.dropna(axis=0, inplace=True)
    elif way == "Replace with mean": df.fillna(df.mean(), inplace=True)
    elif way == "Replace with median": df.fillna(df.median(), inplace=True)
    elif way == "Replace with mode": df.fillna(df.mode(), inplace=True)

    return df


def categorical(df):
    # select method
    way = st.selectbox("Select method to replace categorical values:", ("-----", "Map to dictionary", "One-hot encoding"))

    if way == "Map to dictionary":
        # get categorical columns
        non_num_cols = df.select_dtypes(exclude=np.number)
        # create list that holds mapping dictionaries for each categorical column
        dictionaries = []
        # iterate through the list of columns
        # use enumerate in order to get index 
        for x, y in enumerate(non_num_cols.columns):
            # add dictionary to dictionaries list
            dictionaries.append({j: i for i, j in enumerate(df[y].value_counts(ascending=True).index.tolist())})
            # replace the values using dictionary at index x of column y in the df 
            df[y].replace(dictionaries[x], inplace=True)
    elif way == "One-hot encoding":
        df = pd.get_dummies(df)

    return df


def renamer(df):
    # get columns to rename
    cols_to_rename = st.multiselect("Select columns to rename:",  ("Rename All Columns",) + tuple(vars.df.columns))
    if "Rename All Columns" in cols_to_rename:
        cols_to_rename = list(vars.df.columns)

    names = ""
    cols_dict = {}

    names = st.text_input(f"Enter the names of the columns separated by commas in the following order:\n{str(cols_to_rename)}")
    names_ls = names.replace(" ", "").split(",")

    if st.button("Rename"):
        for i,j in enumerate(cols_to_rename):
            cols_dict[j] = names_ls[i]

        vars.df.rename(columns=cols_dict, inplace=True)

    return df


def dropper(df):
    cols_to_drop = st.multiselect("Select columns to drop:", tuple(df.columns))
    if st.button("Drop"):
        df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def normalize_features(df):
    features = df[df.columns[:-1]]
    scaled_features = StandardScaler().fit_transform(features)
    scaled_features_df = pd.DataFrame(data=scaled_features, columns=[df.columns[:-1]]) 
    scaled_features_df[df.columns[-1]] = df[df.columns[-1]].values
    return scaled_features_df

    
def oversample(df):
    res = df
    choice = st.selectbox("Method to oversample:", ("-----", "RandomOverSampler", "SMOTE"))
    if choice == "RandomOverSampler":
        over_sampler = RandomOverSampler(random_state=42)
    elif choice == "SMOTE":
        over_sampler = SMOTE(random_state=42)

    if choice != "-----":
        features_rus, target_rus = over_sampler.fit_resample(df[df.columns[:-1]], df[df.columns[-1]])
        res = pd.DataFrame(data=features_rus, columns=[df.columns[:-1]])
        res[df.columns[-1]] = target_rus

    return res
        
def undersample(df):
    rus = RandomUnderSampler(sampling_strategy="not minority", random_state=42)
    features_rus, target_rus = rus.fit_resample(df[df.columns[:-1]], df[df.columns[-1]])
    res = pd.DataFrame(data=features_rus, columns=[df.columns[:-1]])
    res[df.columns[-1]] = target_rus


