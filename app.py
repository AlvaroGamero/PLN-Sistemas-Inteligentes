from src.analisis import readExcel, makeCorpus, similitud, opcionesDf

if __name__ == '__main__':

    opcionesDf()
    df = readExcel('tweets.xlsx')
    corpus = makeCorpus(df)
    similitud(corpus, df)
