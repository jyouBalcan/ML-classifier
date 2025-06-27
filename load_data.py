import pandas as pd
import nltk
import re as r
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import LogisticRegression
from bert_model import tokenizer

nltk.download('punkt')
nltk.download('stopwords')


FILEPATH = './report.csv'
COLUMNS = ["Titre", "Description", "Catégorie"]

# only read title and description columns
df = pd.read_csv(FILEPATH, usecols=COLUMNS, encoding = 'utf8')


# cleaning function for each entry
def clean_text(text):
    text = text.lower()
    text = r.sub(r'\d+', '', text)
    text = r.sub(r'[^\w\s]', '', text)
    return str(text).replace('\n', ' ').strip()

print(len(df))
df_list = []
for title, description, cat in zip(df['Titre'], df['Description'], df['Catégorie']):
    
    # clean and tokenize each entry
    entry = str(title) + " " + str(description)
    entry = clean_text(entry)
    entry = tokenizer.tokenize(entry)

    # clean and tokenize category
    cat = clean_text(str(cat))
    cat = tokenizer.tokenize(cat)

    # append to list
    df_list.append([entry, cat])
# convert to list of title + description strings

print (df_list)


