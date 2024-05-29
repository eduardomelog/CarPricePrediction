import streamlit as st
import pandas as pd
import numpy as np
import pickle
import locale
import sklearn

df = pd.read_csv('car_prices.csv')
manufacturers = list(df['manufacturer'].drop_duplicates())
grouped = df.groupby('manufacturer')['description'].unique()
description_dict = grouped.to_dict()

models = sorted(list(df['model'].drop_duplicates()))

st.markdown(
   """
<style>
   body {
       background-color: #ffffff;
   }
</style>
   """,
   unsafe_allow_html=True
)

st.title('Car value estimation site.')

st.subheader('''
                 **Introduce the features of your car on the left side and then click on the "Predict" button.**
                 ''')

model = pickle.load(open('car_price_model_pickle.sav', 'rb'))

manufacturer = st.sidebar.selectbox(
    'Select manufacturer:',
    manufacturers
)

description = st.sidebar.selectbox(
    'Select description:',
    description_dict[manufacturer]
)

mod = st.sidebar.selectbox(
    'Select model:',
    models
)

mileage = st.sidebar.number_input(
    'Enter mileage (km/h):',
    step = 1000
)


if st.button('Predict'):
    X = pd.DataFrame({
        'manufacturer':[manufacturer],
        'description':[description],
        'model':[mod],
        'mileage':[mileage]
                })
    X['prediction'] = model.predict(X)


    #st.write(X)

    price = X['prediction'].values[0]

    # Set the locale to the user's default locale
    locale.setlocale(locale.LC_ALL, '')
    # Define the float number
    float_number = price
    # Convert the float number to a string with thousands separators
    formatted_number = locale.format_string("%d", float_number, grouping=True)

    # Write text with colored background using HTML and include the variable
    st.markdown(f'#### <div style="background-color:#f0f0f0; padding:10px; border-radius:20px;"> The car price based on your features should be: ${formatted_number}</div>', unsafe_allow_html=True)


    st.image(
        'images/' + description +'.jpg',
        #style = 'display: block; margin:auto;'
             )















    

    




    
