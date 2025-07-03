import nltk as n
import re as r
from nltk.corpus import stopwords
from nltk.tag import pos_tag
 
from sklearn.model_selection import train_test_split #for splitting data
from sklearn.linear_model import LogisticRegression #identifying the algorithm used
 
 
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
 
 
n.download('punkt')
n.download('punkt_tab')
n.download('stopwords')
 

# description_one = "Line # 32,204MIX_B,206,207,212,213 no activity since Mon, Jun/23 - 17:00 and those lines are not flagged as down in the Extrusion Lines Screen. The ADC Monitor LN: 32 Last Mixer: Jun 23 16:58 Last Scale: Jun 23 12:34 LN: 206 Last Mixer: Jun 18 06:40 Last Scale: Jun 23 17:54 LN: 207 Last Mixer: Jun 17 17:20 Last Scale: Jun 23 17:34 LN: 212 Last Mixer: Jun 23 11:47 Last Scale: Jun 23 16:29 LN: 213 Last Mixer: Jun 23 11:47 Last Scale: Jun 23 16:15"
def tokenizer(description_one): 
    description_one = description_one.lower()
    description_one = r.sub(r'\d+', '', description_one)
    description_one = r.sub(r'[^\w\s]', '', description_one)
    tokens_one = n.word_tokenize(description_one)
    stop_words_e = stopwords.words('english')
    stop_words_f = stopwords.words('french')
    stop_words = set(stop_words_e + stop_words_f)
    tokens_one = ' '.join(token for token in tokens_one if token not in stop_words)
    return tokens_one

text = u"This is an example in English."
t = tokenizer(text)
print(type(t))
print(t)

text = u"Voici un exemple en français."
t = tokenizer(text)
print(type(t))
print(t)

