import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


with open('model_pickel_tfidf', 'rb') as fe:
    tfidf = pickle.load(fe)

with open('model_pickel_rfc', 'rb') as f:
    model = pickle.load(f)

def preprocess_text(t):
    text = re.sub('[^a-zA-Z]', ' ', t)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    text = ' '.join(text)
    return text

def function(text):
    new_corpus = [text]
    
    new_X_test = tfidf.transform(new_corpus).toarray()
    new_y_pred = model.predict(new_X_test)

    if new_y_pred[0] == 0 :
        return "The Sentiment is Negative 😭"
    elif new_y_pred[0] == 1 :
        return "The Sentiment is Neutral 😃"
    elif new_y_pred[0] == 2 :
        return "The Sentiment is Positive 😁"


st.title("Simple Sentiment Analysis WebApp") 

text = st.text_area("Please Enter your text :")

if st.button("Analyze the Sentiment"):
    result = function(preprocess_text(text))
    st.write(result)
