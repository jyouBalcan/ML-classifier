import pandas as pd
import nltk
import re as r
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk_tokenizer import tokenizer
from balancer import balance


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


# df_list = np.array(len(df))
# categories = np.array(len(df))
df_list = []
categories = []
count = 0
for title, description, category in zip(df['Titre'], df['Description'], df['Catégorie']):

    # clean and tokenize each entry
    entry = str(title) + " " + str(description)
    entry = clean_text(entry)
    entry = tokenizer(entry)

    # clean and tokenize category
    cat = clean_text(str(category))
    cat = tokenizer(cat)

    # append to list
    df_list.append(entry)
    categories.append(cat)
        
    count += 1

    if count % 1000 == 0:
        print(f"Processed {count} entries...")


df_list, categories = balance(df_list, categories)

x_train, x_test, y_train, y_test = train_test_split(df_list, categories, test_size=0.2, random_state = 1000)

# print(type(x_train), type(y_train))
# print(type(x_test), type(y_test))

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

classifier = LogisticRegression(max_iter = 12000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print(f"Model accuracy: {score * 100:.2f}%")


def predict(text):
    text = tokenizer(text)
    return classifier.predict(vectorizer.transform([text]))

while True:
    print(predict(input("ENTER PROBLEM")))