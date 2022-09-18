import pickle as pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

# Page Title
st.title('Fracture Mechanices')

# Loading Model
model = pickle.load(open('final_model.pkl', 'rb'))

# loading Features
features_dict = pickle.load(open('feature.pkl', 'rb'))

# selections
E = st.number_input('Enter the elastic Modulus:')
F = st.number_input('Enter Compressive:')
Pmax = st.number_input('Enter peak load:')
final_crack = st.number_input('Enter crack length:')

if st.button('Predict CTOD'):
    dataset = [E, F, Pmax, final_crack]

df = pd.DataFrame(dataset)
CTOD = model.predict(df.T).flatten().tolist()

st.success(
    f'the CTOD is {CTOD[0]:,.0f}')
