import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smartphone Dashboard", layout="wide")
st.title("Smartphone Dashboard")

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/7ng10dpE/Online-Retail/resolve/main/Smartphones_6M_FINAL.csv"
    df = pd.read_csv(url)
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    return df

df = load_data()

st.success(f"Loaded {len(df):,} rows")
st.dataframe(df.head(), use_container_width=True)
