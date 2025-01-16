import streamlit as st
import pandas as pd

st.title('Machine learning app')

st.info('This is a machine learning app')

with st.expander('Data'):
  st.write('#### Raw data')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**x**')

  x = df.drop(columns = 'species')
  x

  st.write('**y**')
  y = df['species']
  y

with st.expander('Data visualization'):
  st.scatter_chart(data = df, x = 'bill_length_mm', y = 'flipper_length_mm',
            color = 'species')

