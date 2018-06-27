import pandas as pd
from nltk.corpus import stopwords
import re
import string

train_df = pd.read_csv("Data/train.tsv", sep="\t")
test_df = pd.read_csv("Data/test.tsv", sep="\t")

def preprocess_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[%s]' % re.escape(string.digits), '', text)
    text = re.sub('[%s]' % re.escape(' +'), ' ', text)
    text = text.lower()
    text = text.strip()
    stopword = stopwords.words("english")
    text = ' '.join([word for word in text.split() if word not in stopword])
    return text

train_df.Phrase = train_df.Phrase.apply(preprocess_text)
test_df.Phrase = test_df.Phrase.apply(preprocess_text)


