import pickle as pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Page Title
st.title('Fracture Mechanics')

# Loading Model
model = pickle.load(open('first_model.pkl', 'rb'))

# loading scaler
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalerY = pickle.load(open('scalery.pkl', 'rb'))


# data input
E = st.number_input('Enter the Elastic Modulus:')
F = st.number_input('Enter the Compressive Strength:')
Pmax = st.number_input('Enter Peak Load (KN):')
final_crack = st.number_input(
    'Enter crack length (this is the difference between the final and initial crack(mm)):', step=1e-5, format="%.5f")

# make predictions
if st.button('Predict Result'):
    dataset = [E, F, Pmax, final_crack]
    df = pd.DataFrame(dataset)
    df = df.T.values
    polyDF = PolynomialFeatures(
        2, interaction_only=True).fit_transform(df)
    scaledDF = scalerX.transform(polyDF)
    prediction = model.predict(scaledDF)
    prediction = scalerY.inverse_transform(prediction)
    Result = prediction.flatten().tolist()
    st.success(
        f'the CTOD is **{Result[0]:,.5f}** and the KiSC is **{Result[1]:,.5f}** .')
