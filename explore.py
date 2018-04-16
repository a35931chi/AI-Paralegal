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
