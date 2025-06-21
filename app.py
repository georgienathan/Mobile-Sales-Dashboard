import streamlit as st
import pandas as pd
import gdown

@st.cache_data
def load_data():
    file_id = '1LhxxZdladJ5Jn5tX0_fq4-7VgJAGc_5V'  
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'Smartphones_CLEAN.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output, parse_dates=["event_time"])

df = load_data()

st.set_page_config(page_title="Mobile Sales Dashboard", layout="wide")
st.title("📱 Mobile Sales Dashboard")
st.write("✅ Data loaded successfully")
st.dataframe(df.head())
