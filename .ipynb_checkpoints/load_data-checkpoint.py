import pandas as pd

filepath = 'C:/Users/jyou/OneDrive - Balcan Innovations Inc/Documents/ML-classifier/report.csv'

df = pd.read_csv(filepath, encoding = 'utf8')

for row in df:
    print(row)
