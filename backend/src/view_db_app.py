import sqlite3
import pandas as pd
import streamlit as st

st.title("PS: Data is not real, Dummy Data Only")
st.title("AU Small Finance Bank Fraud Cases")


conn = sqlite3.connect("fraud_cases.db")
df = pd.read_sql_query("SELECT * FROM fraud_cases", conn)
conn.close()

st.dataframe(df, use_container_width=True)

