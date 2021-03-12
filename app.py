from itertools import  product
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.analisis import readExcel, makeCorpus, getSentiment, similitud, opcionesDf

if __name__ == '__main__':

    opcionesDf()
    df = readExcel('tweets.xlsx')
    corpus = makeCorpus(df)
    similitud(corpus, df)
