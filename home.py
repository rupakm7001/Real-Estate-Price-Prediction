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
from multiapp import MultiApp
from apps1 import p1,p2


import warnings
warnings.filterwarnings('ignore')



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header(':green[Dashboard] `version 1`')


app1 = MultiApp()

# Add all your application here
st.sidebar.subheader(':orange[Select a Tab]')
app1.add_app("About Dataset",p1.app1)
app1.add_app("House Price Prediction",p2.app1)


app1.run()


