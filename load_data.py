import pandas as pd
import nltk
import re as r
from nltk.corpus import stopwords
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('stopwords')


FILEPATH = 'C:/Users/jyou/OneDrive - Balcan Innovations Inc/Documents/ML-classifier/report.csv'
COLUMNS = ["Titre", "Description"]

df = pd.read_csv(FILEPATH, usecols=COLUMNS, encoding = 'utf8')
# only read title and description columns

# cleaning function for each entry
def clean_text(text):
    text = text.lower()
    text = r.sub(r'\d+', '', text)
    text = r.sub(r'[^\w\s]', '', text)
    return str(text).replace('\n', ' ').strip()


df_list = []
for title, description in zip(df['Titre'], df['Description']):
    entry = str(title) + " " + str(description)
    entry = clean_text(entry)
    
    df_list.append(entry)
# convert to list of title + description strings

print (df_list)



#for titre, description in df:
    #df_list.append(titre + " " + description)

#print(df_list)
#print(df[2:3])  # Print the third row of the DataFrame

