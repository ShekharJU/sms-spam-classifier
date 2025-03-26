import streamlit as st
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib

# Ensure NLTK data is available
nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load vectorizer and model
try:
    tfidf = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    st.success("✅ Model and Vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Streamlit UI
st.title("📧 SMS Spam Classifier")

input_sms = st.text_area("Enter the message to classify")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display
        if result == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
