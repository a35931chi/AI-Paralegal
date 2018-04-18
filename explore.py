import pandas as pd
import os

import nltk   
import codecs
from bs4 import BeautifulSoup

import pymysql
import pymysql.cursors
from sqlalchemy import create_engine, MetaData, String, Integer, Table, Column, ForeignKey

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

def grab_dockets():
    files = []
    #get all .html files in the folder (all docket files are in .html)
    for file in os.listdir('References/'):
        if file.endswith('.html'):
            files.append(os.path.join('References/', file))

    df_docket_texts = pd.DataFrame()
    
    for i in range(len(files)): #gather all docket texts
    #for i in [0, 1]: #for testing purposes
        
        content = codecs.open(files[i], 'r', 'utf-8').read()
        #use beautiful soup to get the case ID
        soup = BeautifulSoup(content, 'lxml')
        case_id = str(soup.find_all('h3'))    
        bookmark1 = case_id.find('CASE #:') + len('CASE #:')
        bookmark2 = case_id.find('</h3>')
        case_id = case_id[bookmark1:bookmark2]

        #use pandas to grab tables in the html files
        docket_tables = pd.read_html(content)

        #error checking: gotta do this because there's different length of docket_list/
        #usually docket texts are in docket_list[3], but not always
        n = 0
        while docket_tables[n].isin(['Docket Text']).sum().sum() == 0:
            #print(n, docket_tables[n].isin(['Docket Text']).sum().sum())
            n += 1
                        
        #print(i, files[i])
        #print(docket_tables[n].head())

        #docket_tables[n] is the docket text table
        new_header = docket_tables[n].iloc[0]
        docket_tables[n] = docket_tables[n][1:]
        docket_tables[n].columns = new_header
        
        docket_tables[n]['#'] = pd.to_numeric(docket_tables[n]['#'],
                                              downcast = 'signed', errors = 'coerce')
        docket_tables[n]['Date Filed'] = pd.to_datetime(docket_tables[n]['Date Filed'])
        docket_tables[n]['Case ID'] = case_id

        df_docket_texts = pd.concat([df_docket_texts, docket_tables[n]])
    #reorder a column
    cols = list(df_docket_texts.columns)
    df_docket_texts = df_docket_texts[[cols[-1]] + cols[:-1]]
    
    print('current docket text table size/shape: {}'.format(df_docket_texts.shape))
    return df_docket_texts
    
#may want to store in SQL database when this gets bigger. pull and update from SQL database
#engine = create_engine('mysql+pymysql://a35931chi:Maggieyi66@localhost/docket_texts')
#temp1.to_sql('combined_zhvi', engine, if_exists = 'replace', index = True)


#text clean and processing
#reference document: http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb
#https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

#standard cleaning method
def clean_standard(sentence):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = ' '.join([i for i in sentence.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    lemmatized = ' '.join(lemma.lemmatize(word, 'v') for word in punc_free.split())
    return lemmatized

def clean_keep_allcaps(sentence):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    new_sent = []
    temp = ''
    split_sent = sentence.split()
    for i in range(len(split_sent)):
        if temp == '' and split_sent[i].isupper():
            temp = split_sent[i]
        elif temp != '' and split_sent[i].isupper():
            temp = temp + '_' + split_sent[i]
        elif temp != '' and ~split_sent[i].isupper():
            new_sent.append(temp)
            new_sent.append(split_sent[i].lower())
            temp = ''
        else:
            new_sent.append(split_sent[i].lower())

    stop_free = ' '.join([i for i in new_sent if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #why isn't this effectively
    lemmatized = ' '.join(lemma.lemmatize(word, 'v') for word in punc_free.split())
    return lemmatized

def gensim_LDA(list_of_text, num_topics = 10, passes = 50):
    #preparing document-term matrix
    # Importing Gensim
    import gensim
    from gensim import corpora

    # Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(list_of_text)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in list_of_text]

    # running LDA model
    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    LDAmodel = LDA(doc_term_matrix, num_topics = num_topics,
                   id2word = dictionary, passes = passes)

    topics = []
    model_output = LDAmodel.print_topics(num_topics = 10, num_words = 5)
    for something in model_output:
        topics.append(' '.join([word for word in something[1].split('"') if word.isalpha()]))
	
    return LDAmodel, topics



# reference: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
def sklearn_LDA(raw_text, no_topics = 10, no_top_words = 5):
    
    def output_topics(model, feature_names, no_top_words, display = False):
        output = []
        for topic_idx, topic in enumerate(model.components_):
            if display:
                print('Topic %d: ' % (topic_idx) + ' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            output.append(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return output

    no_features = 1000

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2,
                                    max_features = no_features,
                                    stop_words = 'english')
    
    tf = tf_vectorizer.fit_transform(raw_text)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    # Run LDA
    lda = LatentDirichletAllocation(n_topics = no_topics, max_iter = 5,
                                    learning_method = 'online',
                                    learning_offset = 50., random_state = 0).fit(tf)
    

    topics = output_topics(lda, tf_feature_names, no_top_words, display = False)

    return lda, topics

def sklearn_NMF(raw_text, no_topics = 10, no_top_words = 5):

    def output_topics(model, feature_names, no_top_words, display = False):
        output = []
        for topic_idx, topic in enumerate(model.components_):
            if display:
                print('Topic %d: ' % (topic_idx) + ' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            output.append(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return output

    no_features = 1000

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2,
                                    max_features = no_features,
                                    stop_words = 'english')
    
    tf = tf_vectorizer.fit_transform(raw_text)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2,
                                       max_features = no_features,
                                       stop_words = 'english')
    tfidf = tfidf_vectorizer.fit_transform(raw_text)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Run NMF
    nmf = NMF(n_components = no_topics, random_state = 1, alpha = 0.1,
              l1_ratio = 0.5, init = 'nndsvd').fit(tfidf)
    
    topics = output_topics(nmf, tfidf_feature_names, no_top_words, display = False)
    
    return nmf, topics


if __name__ == '__main__':
    #current docket text table size/shape: (721, 4), 2018-04-18
    df = grab_dockets()
    
    #not changed
    doc_original = list(df['Docket Text'])
    doc_original_split = [doc.split() for doc in doc_original]
    #normal cleaning: lowercase, remove stopwords, punctuations, lemmatized
    doc_standard = [clean_standard(doc) for doc in doc_original]
    doc_standard_split = [clean_standard(doc).split() for doc in doc_original]
    #keep all caps cleaning: everything standard, but keeping all caps words/phrases
    doc_keep_allcaps = [clean_keep_allcaps(doc) for doc in doc_original]
    doc_keep_allcaps_split = [clean_keep_allcaps(doc).split() for doc in doc_original]
    
    
    if False: #output docket text and transformations into .csv
        #output the entired docket text
        df = grab_dockets()
        #output the cleaned/transformed docket text
        df['standard_clean'] = doc_standard
        df['keep_allcaps_clean'] = doc_keep_allcaps

        df.to_csv('docket_texts.csv', index = False)
        


    '''
    #do some comparisions:
    print(doc_original[-1])
    print(doc_keep_allcaps[-1])
    bookmark = input('bookmark')
    '''

    files = [doc_original, doc_standard, doc_keep_allcaps]
    files_split = [doc_original_split, doc_standard_split, doc_keep_allcaps_split]
    results = {}
    for i in range(len(files)):
        
        #output the topic output by different models
        #sklearn LDA 
        SKLDA_model, SKLDA_topics = sklearn_LDA(files[i]) #uncleaned text
        results[('SKLDA', i)] = SKLDA_topics
        #print('SKLEARN LDA')
        #print(SKLDA_topics)

        #sklearn MNF
        SKNMF_model, SKNMF_topics = sklearn_NMF(files[i])
        results[('SKNMF', i)] = SKNMF_topics
        #print('SKLEARN NMF')
        #print(SKNMF_topics)

        #gensim LDA
        GLDA_model, GLDA_topics = gensim_LDA(files_split[i])
        results[('GLDA', i)] = GLDA_topics
        #print('GENSIM LDA')
        #print(GLDA_topics)
    
    
    pd.DataFrame(results).to_csv('topics.csv')
