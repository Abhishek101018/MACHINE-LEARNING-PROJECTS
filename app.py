import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import sklearn


def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenize

    y = []
    for i in text:
        if i.isalnum():  # remove special character
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # remove stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # stemming

    return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms=st.text_input("Enter the message")


if st.button('Predict'):
    # 1.preprocess
    transformed_sms=transform_text(input_sms)
    # 2.vectorize
    vector_input=tfidf.transform([transformed_sms])
    # 3.predict
    result=model.predict(vector_input)[0]
    # 4.display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")