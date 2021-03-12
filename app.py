from itertools import combinations, product

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.analisis import readExcel, makeCorpus, makeQuery, getSentiment

if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = readExcel('tweets.xlsx')
    corpus = makeCorpus(df)

    l1 = list(range(len(corpus)-1))
    pairs = list(product(l1, [len(corpus)-1]))
    results_tfidf = [cosine_similarity([corpus[a_index]], [corpus[b_index]]) for (a_index, b_index) in pairs]

    listaTop = sorted(zip(results_tfidf, pairs), reverse=True)[:5]

    listaBot = sorted(zip(results_tfidf, pairs), reverse=False)[:5]

    for x in listaTop:
        # print(df.loc[[x[1][0]]].to_string(index = False))

        text = df.iloc[[x[1][0]],1].to_string(index = False)
        print(text)
        getSentiment(text)
print('-------------------------------')
for x in listaBot:
        # print(df.loc[[x[1][0]]].to_string(index = False))

        text = df.iloc[[x[1][0]],1].to_string(index = False)
        print(text)
        getSentiment(text)

