import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder # Imports simplified
from tensorflow.keras.models import load_model


# load the imdb dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the pretrained model with relu activation
model = load_model('./model/rnn_model.h5')

# function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    return padded_review

# prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('Imdb Review Sentiment Analysis App')
st.write('Enter a review to predict its sentiment.')

# user input
user_input = st.text_area('Enter your review here:')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    # Predict sentiment
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # display the result
    st.write(f'Predicted Sentiment: {sentiment}')
    st.write(f'Confidence Score: {prediction[0][0]}')
else:
    st.write('Please enter a review to classify.')