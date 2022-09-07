import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import nltk
from collections import Counter
import heapq
import re
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from transformers import pipeline
from summarize import *

#Load model
hub_model_id = "shivaniNK8/t5-small-finetuned-cnn-news"
summarizer = pipeline("summarization", model=hub_model_id)


#Create header
st.write("""# HIGHLIGHTS! \n ### A News Summarizer""")
st.write("Provide a news article and get a summary within seconds! ")

# Image
image = Image.open('newspaper.jpeg')
st.image(image)


#Create and name sidebar
st.sidebar.header('Select summary parameters')
with st.sidebar.form("input_form"):
    st.write('Select summary length for extractive summary')
    max_sentences = st.slider('Summary Length', 1, 10, step=1, value=3)
    st.write('Select word limits for abstractive summary')
    max_words = st.slider('Max words', 50, 500, step=10, value=200)
    min_words = st.slider('Min words', 10, 450, step=10, value=100)

    submit_button = st.form_submit_button("Summarize!")
    

article = st.text_area(label = "Enter the article you want to summarize", height = 300, value = "Enter Article Body Here")



news_summarizer = NewsSummarization()


if submit_button:
    st.write("## Extractive Summary")
    ex_summary = news_summarizer.extractive_summary(article, num_sentences = max_sentences)
    st.write(ex_summary)

    summary = summarizer(article, max_length = max_words, min_length = min_words, do_sample = False)
    abs_summary = summary[0]['summary_text']
    st.write("## Abstractive Summary")
    st.write(abs_summary)

with st.sidebar.expander("More About Summarization"):
     st.markdown("""
        In extractive summarization, we identify important sentences from the article and make a summary by selecting the most important sentences. <br>

        Whereas, for abstractive summarization the model understands the context and generates a summary with the important points with new phrases and language. 
        Abstractive summarization is more similar to the way a human summarizes any content. A person might read the entire document, 
        remember a few key points and while writing the summary, will make new sentences that include these points. Abstractive summarization follows the same concept.

     """)
     
    



