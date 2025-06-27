from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

text = "This is an example in English."
tokens = tokenizer.tokenize(text)
print(tokens)

text = u"Voici un exemple en français."
tokens = tokenizer.tokenize(text)
print(tokens)