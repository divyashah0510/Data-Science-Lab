import streamlit as st
from transformers import pipeline

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Transformer-based Sentiment Analysis",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title("🤖 Transformer-based Sentiment Analysis using Huggingface")

# Description
st.write("""
This application analyzes the sentiment of input text using a pre-trained BERT model from the Huggingface Transformers library. 
Enter some text below and click 'Analyze Sentiment' to get the result.
""")

# Load the Huggingface sentiment analysis pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_pipeline()

# Input text box for the user
user_input = st.text_area("Enter text to analyze sentiment", height=150)

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Perform sentiment analysis
        result = sentiment_pipeline(user_input)

        # Display the results dynamically
        st.subheader("Sentiment Analysis Result")
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']
        
        if sentiment_label == "POSITIVE":
            st.success(f"**Sentiment: {sentiment_label}** (Score: {sentiment_score:.2f})")
        elif sentiment_label == "NEGATIVE":
            st.error(f"**Sentiment: {sentiment_label}** (Score: {sentiment_score:.2f})")
        else:
            st.warning(f"**Sentiment: {sentiment_label}** (Score: {sentiment_score:.2f})")
    else:
        st.warning("Please enter some text to analyze.")

# Add a footer
st.markdown("""
---
**Note**: This application uses a pre-trained BERT model from the Huggingface library, making it easy to perform sentiment analysis on various texts.
""")
