import pandas as pd
import os

import nltk   
import codecs
from bs4 import BeautifulSoup

import pymysql
import pymysql.cursors
from sqlalchemy import create_engine, MetaData, String, Integer, Table, Column, ForeignKey


files = []
#get all .html files in the folder
for file in os.listdir('References/'):
    if file.endswith('.html'):
        files.append(os.path.join('References/', file))


#use beautiful soup to get the case ID
df_all = pd.DataFrame()
for i in range(len(files)): #gather all docket texts
#for i in [0, 1]: #for testing purposes
    content = codecs.open(files[i], 'r', 'utf-8').read()
    soup = BeautifulSoup(content, 'lxml')
    case_id = str(soup.find_all('h3'))
    
    bookmark1 = case_id.find('CASE #:') + len('CASE #:')
    bookmark2 = case_id.find('</h3>')
    case_id = case_id[bookmark1:bookmark2]

    docket_list = pd.read_html(content)

    #error checking: gotta do this because there's different length of docket_list
    n = 0
    while docket_list[n].isin(['Docket Text']).sum().sum() == 0:
        #print(n, docket_list[n].isin(['Docket Text']).sum().sum())
        n += 1
    
    #print(i, files[i])
    #print(docket_list[n].head())
    
    new_header = docket_list[n].iloc[0]
    docket_list[n] = docket_list[n][1:]
    docket_list[n].columns = new_header
    
    docket_list[n]['#'] = pd.to_numeric(docket_list[n]['#'], downcast='signed', errors = 'coerce')
    docket_list[n]['Date Filed'] = pd.to_datetime(docket_list[n]['Date Filed'])
    docket_list[n]['Case ID'] = case_id

    df_all = pd.concat([df_all, docket_list[n]])
    
print(df_all.shape)
    

#engine = create_engine('mysql+pymysql://a35931chi:Maggieyi66@localhost/docket_texts')
#temp1.to_sql('combined_zhvi', engine, if_exists = 'replace', index = True)


''' output to .csv
print(docket_list[3])
for i in range(len(docket_list)):
    docket_list[i].to_csv('{} [{}].csv'.format(case_id, i))
'''

#reference document: http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb
#https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

#clean and processing
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean_method1(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

full_doc = list(df_all['Docket Text'])

#note that when we cleaned with normal 
doc_clean1 = [clean_method1(doc).split() for doc in full_doc]

#do some comparisions:
print(full_doc[-1])
print(doc_clean1[-1])

def gensim_LDA(list_of_text, num_topics = 3, passes = 50):
    #preparing document-term matrix
    # Importing Gensim
    import gensim
    from gensim import corpora

    # Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean1)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean1]

    # running LDA model
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary,
                   passes = passes)

    #Results, displaying 3 topics
    print(ldamodel.print_topics(num_topics=3, num_words=3))

    pass #ideally I want to pass back the model

# reference: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

def sklearn_LDA():
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import NMF, LatentDirichletAllocation

    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    dataset = fetch_20newsgroups(shuffle = True, random_state = 1,
                                 remove = ('headers', 'footers', 'quotes'))
    documents = dataset.data

    no_features = 1000

    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 20

    # Run NMF
    nmf = NMF(n_components = no_topics, random_state = 1, alpha = 0.1,
              l1_ratio = 0.5, init = 'nndsvd').fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_topics = no_topics, max_iter = 5,
                                    learning_method = 'online',
                                    learning_offset = 50., random_state = 0).fit(tf)

    no_top_words = 10
    display_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(lda, tf_feature_names, no_top_words)

    pass

sklearn_LDA()
