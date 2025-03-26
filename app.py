import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

tfidf = joblib.load('vectorizer.pkl')  # Make sure this file exists
model = joblib.load('model.pkl')       # Make sure this file exists


# Ensure NLTK data is available
nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuations
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
