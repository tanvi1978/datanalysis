import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.express

# decision tree
from sklearn.tree import export_graphviz
from io import StringIO
from sklearn import tree

import vars


def app():
    st.title("Visualize Data")
    st.write("Select the plots you want to plot. If you chose the decision tree classifier for predicition, you can vizualize the decision tree here.")

    
    plot_options = ["Correlation Heatmap", "Scatter Plot", "3D Scatter Plot", "Line Chart", "Area Chart", "Count Plot", "Pie Chart", "Box Plot", "Subplot"]
    if vars.dtree_bool:
        plot_options.append("Decision Tree")


    plot_list = st.multiselect("Select the charts/plots", plot_options)

    if 'Correlation Heatmap' in plot_list:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10,10))
        sns.heatmap(vars.df.corr(), annot=True)
        st.pyplot()

    if 'Line Chart' in plot_list:
        st.subheader("Line Chart")
        st.line_chart(vars.df)
    
    if 'Area Chart' in plot_list:
        st.subheader("Area Chart")
        st.area_chart(vars.df)

    if 'Count Plot' in plot_list:
        st.subheader("Count Plot")
        plt.figure(figsize=(10,10))
        choice = st.selectbox("Select the columns for count plot:", tuple(vars.df.columns))
        sns.countplot(x = choice, data=vars.df)
        st.pyplot()
    
    if 'Pie Chart' in plot_list:
        st.subheader("Pie Chart")
        plt.figure(figsize=(10,10))
        choice = st.selectbox("Select the column for pie chart:", tuple(vars.df.columns))
        plt.pie(vars.df[choice].value_counts(), labels=vars.df[choice].value_counts().index, autopct="%1.2f%%")
        st.pyplot()

    if 'Box Plot' in plot_list:
        st.subheader("Box Plot")
        columns = st.selectbox("Select the columns for box plot:", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
        plt.figure(figsize=(10,10))
        sns.boxplot(vars.df[columns])
        st.pyplot()

    if 'Scatter Plot' in plot_list:
        st.subheader("Scatter Plot")
        c1 = st.selectbox("Select the column for x-axis of scatter plot:", tuple(vars.df.columns))
        c2 = st.selectbox("Select the column for y-axis of scatter plot:", tuple(vars.df.columns))
        st.write(c1, c2)
        plt.figure(figsize=(15,10))
        plt.scatter(vars.df[c1], vars.df[c2])
        st.pyplot()

    if '3D Scatter Plot' in plot_list:
        st.subheader("3D Scatter Plot")
        x = st.selectbox("Select the column for x-axis of scatter plot:", tuple(vars.df.columns))
        y = st.selectbox("Select the column for y-axis of scatter plot:", tuple(vars.df.columns))
        z = st.selectbox("Select the column for z-axis of scatter plot:", tuple(vars.df.columns))
        if st.button("Plot"):
            fig =  plotly.express.scatter_3d(vars.df, x=x, y=y, z=z, color=vars.df[vars.df.columns[-1]])
            fig.show()

    if 'Decision Tree' in plot_list:
        dot_data = StringIO()
        dot_data = tree.export_graphviz(decision_tree = vars.info[0], max_depth = 3, out_file = None, filled = True, rounded = True, feature_names =vars.df.columns[:-1], class_names = ['0', '1'])
        st.graphviz_chart(dot_data)

