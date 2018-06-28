import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import re
import string
from wordcloud import WordCloud

train_df = pd.read_csv("Data/train.tsv", sep="\t")
test_df = pd.read_csv("Data/test.tsv", sep="\t")

def preprocess_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[%s]' % re.escape(string.digits), '', text)
    text = re.sub('[%s]' % re.escape(' +'), ' ', text)
    text = text.lower()
    text = text.strip()

    ## Remove stopwords
    stopword = stopwords.words("english")
    text = ' '.join([word for word in text.split() if word not in stopword])

    ## Convert to root words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word.strip()) for word in text.split(' ')])

    return str(text)

train_df.Phrase = train_df.Phrase.apply(preprocess_text)
test_df.Phrase = test_df.Phrase.apply(preprocess_text)

train_df["phrase_length"] = train_df.Phrase.apply(lambda x: len(x.split()))
test_df["phrase_length"] = test_df.Phrase.apply(lambda x: len(x.split()))

#cloud = WordCloud(width=800, height=800, background_color="white", stopwords=stopwords.words("english"), min_font_size=10).generate(train_df.Phrase)

tokenizer = TweetTokenizer()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(train_df['Phrase'].values) + list(test_df['Phrase'].values)
vectorizer.fit(full_text)

train_vectorized = vectorizer.transform(train_df['Phrase'])
test_vectorized = vectorizer.transform(test_df['Phrase'])

y = train_df['Sentiment']

logreg = LogisticRegression()
ovr = OneVsRestClassifier(logreg)

ovr.fit(train_vectorized, y)

scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
