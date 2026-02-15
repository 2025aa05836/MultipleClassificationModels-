import streamlit as st
import pandas as pd

# st.write('Hello World')

st.set_page_config(page_title = 'File uploader')

df = st.file_uploader(label = 'upload your dataset')

if df:
    df = pd.read_csv(df)
    st.write(df.head())
