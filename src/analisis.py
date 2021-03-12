import re
import string
import nltk as nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from textblob import TextBlob

def readExcel(path):
    df = pd.read_excel(path)
    df = df[["From-User","Text"]]
    return df

def makeCorpus(df):

    corpus = []
    for row in df['Text']:
        corpus.append(limpieza(row))

    corpus.append(limpieza(makeQuery()))

    stop_es=stopwords.words('spanish')
    cv_tfidf = TfidfVectorizer(analyzer='word', stop_words = stop_es)
    X_tfidf = cv_tfidf.fit_transform(corpus).toarray()
    # print(pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names()))
    return X_tfidf

def makeQuery():
    query = input("Introduce aqui tu consulta: ")

    return query


def limpieza(texto):

    spanish_stemmer = nltk.SnowballStemmer('spanish')

    clean_text = re.sub("[%s]" % re.escape(string.punctuation), " ", texto)
    clean_text= clean_text.lower()
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    clean_text = re.sub('@', ' ', clean_text)
    clean_text = re.sub('http.*', ' ', clean_text)
    tokenized = word_tokenize(clean_text)

    for j in range(len(tokenized)):
        tokenized[j]=format(spanish_stemmer.stem(tokenized[j]))
    stemmed = ' '.join(tokenized)

    return stemmed


def getSentiment(textInput):
    # print(textInput)
    analysis = TextBlob(textInput)
    language = analysis.detect_language()
    if language != 'en':
        analysis= analysis.translate(to='en')
    # print(analysis.sentiment)
    analysisPol = analysis.sentiment.polarity
    analysisSub = analysis.sentiment.subjectivity
    print(f'Tiene una polaridad de {analysisPol} y una subjectibidad de {analysisSub}')
    return None
