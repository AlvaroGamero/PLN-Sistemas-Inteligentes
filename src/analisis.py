import re
import string
from itertools import product
import nltk as nltk
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from operator import itemgetter, attrgetter
nltk.download('stopwords')

def readExcel(path):
    df = pd.read_excel(path)
    df = df[["From-User", "Text"]]
    return df

def makeCorpus(df):
    corpus = []
    for row in df['Text']:
        corpus.append(limpieza(row))
    corpus.append(limpieza(makeQuery()))
    stop_es = stopwords.words('spanish')
    cv_tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_es)
    X_tfidf = cv_tfidf.fit_transform(corpus).toarray()
    # print(pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names()))
    return X_tfidf

def makeQuery():
    query = input("\nIntroduce aquí tu consulta: ")
    return query

def limpieza(texto):
    spanish_stemmer = nltk.SnowballStemmer('spanish')
    clean_text = re.sub("[%s]" % re.escape(string.punctuation), " ", texto)
    clean_text = clean_text.lower()
    clean_text = re.sub('\w*\d\w*', ' ', clean_text)
    clean_text = re.sub('@', ' ', clean_text)
    clean_text = re.sub('http.*', ' ', clean_text)
    tokenized = word_tokenize(clean_text)
    for j in range(len(tokenized)):
        tokenized[j] = format(spanish_stemmer.stem(tokenized[j]))
    stemmed = ' '.join(tokenized)
    return stemmed

def getSentiment(textInput):
    analysis = TextBlob(textInput)
    try:
        language = analysis.detect_language()
        # La traduccion peta en algunas querys por ejemplo futbol por eso la he comentado
        if language != 'en':
            analysis = analysis.translate(to='en')
    except:
        print("La traducción no ha funcionado")
    analysisPol = analysis.sentiment.polarity
    analysisSub = analysis.sentiment.subjectivity
    print(f'Tiene una polaridad de {analysisPol} y una subjectibidad de {analysisSub}')
    return analysisPol

def similitud(corpus, df):
    l1 = list(range(len(corpus) - 1))
    pairs = list(product(l1, [len(corpus) - 1]))
    results_tfidf = [cosine_similarity([corpus[a_index]], [corpus[b_index]]) for (a_index, b_index) in pairs]
    listaTop = sorted(zip(results_tfidf, pairs), reverse=True)[:5]
    listaBot = sorted(zip(results_tfidf, pairs), reverse=False)[:5]
    listaSentTop = []
    listaSentBot = []
    i=1
    print("\n-------------------------TWEETS SIMILARES------------------------------\n")
    print("TOP 5 tweets similares: ")
    for x in listaTop:
        text = df.iloc[[x[1][0]], 1].to_string(index=False)
        print("Tweet",i,":",text)
        listaSentTop.append((getSentiment(text), text, i))
        print()
        i+=1
    listaSentTopSorted = sorted(listaSentTop, key=itemgetter(0), reverse=True)
    print("\nTop 5 Tweets similares ordenados por polaridad")
    for x in listaSentTopSorted:
        print("Tweet ",x[2], ": Polaridad: ", x[0])

    print("\n-------------------------TWEETS DISTINTOS------------------------------\n")
    print("TOP 5 tweets distintos: ")
    i=1
    for x in listaBot:
        text = df.iloc[[x[1][0]], 1].to_string(index=False)
        print("Tweet",i,":",text)
        listaSentBot.append((getSentiment(text), text, i))
        print()
        i+=1
        listaSentBotSorted = sorted(listaSentBot, key=itemgetter(0), reverse=True)
    print("\nTop 5 Tweets distintos ordenados por polaridad")
    for x in listaSentBotSorted:
        print("Tweet ",x[2], ": Polaridad: ", x[0])
    return None

def opcionesDf():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
