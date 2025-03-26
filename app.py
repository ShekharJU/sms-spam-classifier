import streamlit as st
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import traceback

# Ensure NLTK data is available
nltk_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_path, exist_ok=True)
nltk.data.path.append(nltk_path)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Check File Existence and Path
print("Vectorizer exists:", os.path.isfile('vectorizer.pkl'))
print("Model exists:", os.path.isfile('model.pkl'))

if os.path.isfile('vectorizer.pkl'):
    print("Vectorizer Size (in bytes):", os.path.getsize('vectorizer.pkl'))

if os.path.isfile('model.pkl'):
    print("Model Size (in bytes):", os.path.getsize('model.pkl'))

print("Current Directory:", os.getcwd())

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return " ".join(ps.stem(word) for word in tokens if word.isalnum() and word not in stopwords.words('english'))

# Load vectorizer and model
try:
    tfidf = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    st.success("✅ Model and Vectorizer loaded successfully.")
    print("Files loaded successfully using joblib.")
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()
except Exception as e:
    print("Error loading model/vectorizer:", e)
    traceback.print_exc()
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
