import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import plotly.express as px

# Set page config for a wider layout
st.set_page_config(layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Ollama
llm = Ollama(model="tinyllama")       #mixtral or mistral or other models.

def analyze_data(df, column):
    """Perform basic analysis on a dataframe column."""
    if df[column].dtype == 'object':
        return f"Most common value: {df[column].mode().values[0]}"
    else:
        return f"Mean: {df[column].mean():.2f}, Median: {df[column].median():.2f}"

st.title("🚀 Advanced Data Analysis App")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Create SmartDataframe
    smart_df = SmartDataframe(data, config={"llm": llm})
    
    # Display basic info about the dataset
    st.sidebar.info(f"📊 Dataset Info:\nRows: {data.shape[0]}\nColumns: {data.shape[1]}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(data.head())
        
        st.subheader("🔍 Data Exploration")
        if st.checkbox("Show data types"):
            st.write(data.dtypes)
        if st.checkbox("Show summary statistics"):
            st.write(data.describe())
        if st.checkbox("Show column names"):
            st.write(data.columns)
    
    with col2:
        st.subheader("📊 Quick Visualization")
        viz_column = st.selectbox("Select a column to visualize", data.select_dtypes(include=['int64', 'float64']).columns)
        fig = px.histogram(data, x=viz_column)
        st.plotly_chart(fig)
    
    st.subheader("🧠 Smart Data Analysis")
    analysis_question = st.text_input("Ask a question about your data (e.g., 'What is the average price?', 'How many unique categories are there?')")
    if analysis_question:
        with st.spinner("Analyzing..."):
            try:
                result = smart_df.chat(analysis_question)
                st.success(f"Answer: {result}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    st.subheader("🔢 Traditional Data Analysis")
    column = st.selectbox("Select a column to analyze", data.columns)
    if st.button("Analyze"):
        result = analyze_data(data, column)
        st.info(result)
    
    st.subheader("🔎 Custom Query")
    query = st.text_input("Enter a pandas query (e.g., 'column > 5')")
    if query:
        try:
            filtered_data = data.query(query)
            st.write(filtered_data)
        except Exception as e:
            st.error(f"Error in query: {str(e)}")
else:
    st.info("👆 Upload a CSV file to get started!")