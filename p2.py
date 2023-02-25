import streamlit as st
import pandas as pd
import matplotlib as plt
import plotly.graph_objs as go
import numpy as np
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

def app1():
    model = pickle.load(open('home_prices_model.pickle','rb'))
    df = pd.read_csv("data1.csv")
    d1=df.copy()
    df1 = pd.read_csv("data2.csv")
    d2 = df1.copy()
    
    st.title(":green[House Price Prediction]")

    # X = df1.drop(['prices_lakh'],axis='columns')
    # y = df1.prices_lakh

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

    # from sklearn.linear_model import LinearRegression
    # lr_clf = LinearRegression()
    # lr_clf.fit(X_train,y_train)

    def predict_price(location,sqft,bath,bhk,price_per_sqft):  

        # loc_index = np.where(X.columns==location)[0][0]

        x = np.zeros(27)
        x[0] = sqft
        x[1] = bhk
        x[2] = bath
        x[7] = price_per_sqft

        # if loc_index >= 0:
            # x[loc_index] = 1
        return model.predict([x])[0]

    st.subheader('Please enter the required details:')
    location = st.selectbox("Location",df['Location'].unique())
    sqft = st.text_input("Sq-ft area","")
    bath = st.text_input("Number of Bathroom","")
    bhk = st.text_input("Number of BHK","")
    price_per_sqft = st.text_input("Price per sqft", "")

    result=""

    # import random
    # result= random.randint(65,100)
 
    if st.button("House Price in Lakhs"):
        result=predict_price(location,sqft,bath,bhk,price_per_sqft)
    st.success(f'The output is {result}')

          