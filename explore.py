import pandas as pd

import nltk   
import codecs
from bs4 import BeautifulSoup
case_id = '117-cv-01175-ALC'
location = 'References/CIVILDOCKET_' + case_id + '.html'
content = codecs.open(location, 'r', 'utf-8').read()

docket_list = pd.read_html(content)
new_header = docket_list[3].iloc[0]
docket_list[3] = docket_list[3][1:]
docket_list[3].columns = new_header

docket_list[3]['#'] = pd.to_numeric(docket_list[3]['#'], downcast='signed', errors = 'coerce')
docket_list[3]['Date Filed'] = pd.to_datetime(docket_list[3]['Date Filed'])

print(docket_list[3])
for i in range(len(docket_list)):
    docket_list[i].to_csv('{} [{}].csv'.format(case_id, i))
