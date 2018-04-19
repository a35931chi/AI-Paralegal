from explore import grab_dockets, clean_standard, clean_keep_allcaps

import pandas as pd

#first follow along an NLP exercise to figure out the big picture
#http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb




if False: #this is to apply to my own learning
    df = grab_dockets

    doc_original = list(df['Docket Text'])
    doc_original_split = [doc.split() for doc in doc_original]
    #normal cleaning: lowercase, remove stopwords, punctuations, lemmatized
    doc_standard = [clean_standard(doc) for doc in doc_original]
    doc_standard_split = [clean_standard(doc).split() for doc in doc_original]
    #keep all caps cleaning: everything standard, but keeping all caps words/phrases
    doc_keep_allcaps = [clean_keep_allcaps(doc) for doc in doc_original]
    doc_keep_allcaps_split = [clean_keep_allcaps(doc).split() for doc in doc_original]
