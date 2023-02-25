import streamlit as st
import pandas as pd
import matplotlib as plt
import plotly.graph_objs as go
import numpy as np
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

def app1():
    df = pd.read_csv("data1.csv")
    data=df.copy()
    
    st.sidebar.subheader(':orange[Data Analysis]')
    data_inf = st.sidebar.selectbox('Information about the data', ('No','Yes')) 
   
    if data_inf == "No":
            st.title(":violet[About Dataset]")
            st.subheader("We have perform webscrapping on the real estate properties data from 99acres website.")
            st.subheader("We have used linear regression model to predict the house Prices.")

    if data_inf == "Yes":
     st.title(":violet[Data Analysis]")
     st.write("### :blue[Enter the number of rows to view]")
     rows = st.number_input("", min_value=0,value=5)
     if rows > 0:
         st.dataframe(df.head(rows))
         st.subheader(":blue[Data Description]")
         st.write(df.describe())
    

    if data_inf == "Yes":
        st.subheader(":blue[Data Correlation]")
        st.write(df.corr())    
        
        cols = df.columns.drop(['Location','area_ Built-up Area','area_ Carpet Area','area_ Plot Area','area_ Super built-up Area'])
        
        st.markdown('### :blue[Total sqft vs Prices(lakhs)]')
        fig = px.scatter(df, x='total_sqft', y="prices_lakh")
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)
       

        st.markdown('### :blue[Pair plot of Data]')
        fig = sns.pairplot(df[cols],hue="bhk")
        st.pyplot(fig)

        st.markdown('### :blue[Regplot of Price per sqft vs Prices]')
        fig = px.scatter(df,x = "price_per_sqft", y = "prices_lakh")
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)
        
        # fig = px.box(data, x="year", y="Price")
        # fig.update_yaxes(tickfont=dict(size=8))
        # fig.update_xaxes(tickfont=dict(size=8))
        # fig.update_layout(height=600)
        # st.plotly_chart(fig,use_container_width=True)



        
