import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 
import contractions
import pickle
import re

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

@st.cache
def yukDecontract(kalimat):
  kalimat = re.sub(r"n t "," not ",kalimat)
  kalimat = re.sub(r"\bre\b","are",kalimat)
  kalimat = re.sub(r"\bs\b","is",kalimat)
  kalimat = re.sub(r"\bd\b","would",kalimat)
  kalimat = re.sub(r"\bll\b","will",kalimat)
  kalimat = re.sub(r"\bve\b","have",kalimat)
  kalimat = re.sub(r"\bm\b","am",kalimat)
  kalimat = re.sub(r"\bw\b","not",kalimat)
  kalimat = re.sub(r"\bdunno\b","do not know",kalimat)
  return kalimat

@st.cache
def yukDecontractPakaiLibrary(sentence):
  expanded_word = []
  for word in sentence.split():
    expanded_word.append(contractions.fix(word))
  expanded = ' '.join(expanded_word)
  return expanded

@st.cache
def yukLemmatize(sentence):
  lemmatized_word = []
  for word in sentence.split():
    lemmatized_word.append(lemmatizer.lemmatize(word))
  lemmatized = ' '.join(lemmatized_word)
  return lemmatized

@st.cache
def yukBersihin(sentence):
  sentence = yukDecontractPakaiLibrary(sentence)
  sentence = yukDecontract(sentence)
  sentence = yukLemmatize(sentence)
  sentence = re.sub(r" url ","",sentence)
  sentence = re.sub(r" fn ","",sentence)
  sentence = re.sub(r" n ","",sentence)
  sentence = re.sub(r" href ","",sentence)
  sentence = re.sub(r" http ","",sentence)
  sentence = re.sub(r" www ","",sentence)
  return sentence

def main():
  st.title('Emotion Prediction From Text')

  text = st.text_input('Text')

  if st.button("Predict"):
    text = yukBersihin(text)

    # load tfidf
    picklefile = open("./pickles/tfidf.pkl", "rb")
    tfidf = pickle.load(picklefile)
    picklefile.close()

    # apply tfidf
    vectorized_text = tfidf.transform(np.array([text]))
    X = pd.DataFrame(vectorized_text.toarray())
    X.columns = tfidf.get_feature_names_out()

    # load model
    picklefile = open("./pickles/svm-model.pkl", "rb")
    model = pickle.load(picklefile)
    picklefile.close()

    # predict
    pred = model.predict(X)[0]

    if pred == 0:
      st.write('Result: Anger')
    elif pred == 1:
      st.write('Result: Fear')
    elif pred == 2:
      st.write('Result: Joy')
    elif pred == 3:
      st.write('Result: Love')
    elif pred == 4:
      st.write('Result: Sadness')
    elif pred == 5:
      st.write('Result: Surpise')


if __name__ == '__main__':
  main()