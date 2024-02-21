# %%
import pandas as pd
import numpy as np


# %%
jamaal_df = pd.read_csv("COVID19_data.csv")
pd.set_option('display.max_colwidth', 1)
jamaal_df

# %%
jamaal_df = jamaal_df.drop("user", axis=1)
jamaal_df

# %%
import string
jamaal_df['text'] = jamaal_df['text'].str.split(':', n=1).str[1]
jamaal_df['text'] = jamaal_df['text'].str.replace('@', '')
jamaal_df['text'] = jamaal_df['text'].str.replace('#', '')
jamaal_df['text'] = jamaal_df['text'].str.lower()
jamaal_df['text'] = jamaal_df['text'].str.replace("\d+", "", regex=True)
jamaal_df['text'] = jamaal_df['text'].str.translate(str.maketrans('', '', string.punctuation))
# jamaal_df['text'] = jamaal_df['text'].str.replace('…', '')
# jamaal_df['text'] = jamaal_df['text'].str.replace('“', '')



# %%
missing_rows = jamaal_df.isnull().sum(axis=1)
num_missing_rows = len(missing_rows[missing_rows > 0])
print(f"Number of missing rows: {num_missing_rows}")

# %%
jamaal_df.dropna(inplace=True)

# %%
# Drop non-ascii characters
jamaal_df['text'] = jamaal_df['text'].apply(
    lambda x: ''.join([i if ord(i) < 128 else '' for i in x]))

# %%
jamaal_df

# %%
# Display the first few rows of the dataset
jamaal_df.head()

# Display the summary statistics of the dataset
jamaal_df.describe()

# Count the number of rows and columns in the dataset
num_rows, num_cols = jamaal_df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

# %%
jamaal_df['tweet_len'] = jamaal_df['text'].apply(lambda x: len(x))
jamaal_df

# %%
from gensim.models import Word2Vec, KeyedVectors

pretrainedpath = "GoogleNews-vectors-negative300.bin"
pretrained_w2v = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True)


# %%
positive_words = pd.read_csv("positive-words.txt", header=None)
positive_words

# %%
negative_words = pd.read_csv(
    "negative-words.txt", header=None, encoding='ISO-8859-1')
negative_words

# %%
positive_words = pd.read_csv("positive-words.txt", header=None)
negative_words = pd.read_csv("negative-words.txt", header=None, encoding='ISO-8859-1')

jamaal_df['positive_count'] = jamaal_df['text'].apply(lambda x: sum(1 for word in x.split() if word in positive_words[0].values))
jamaal_df['negative_count'] = jamaal_df['text'].apply(lambda x: sum(1 for word in x.split() if word in negative_words[0].values))

jamaal_df['positive_percentage'] = jamaal_df['positive_count'] / jamaal_df['tweet_len']
jamaal_df['negative_percentage'] = jamaal_df['negative_count'] / jamaal_df['tweet_len']

jamaal_df.drop(['positive_count', 'negative_count'], axis=1, inplace=True)


jamaal_df


# %%
jamaal_df.loc[jamaal_df['positive_percentage'] > jamaal_df['negative_percentage'], 'predicted_sentiment_score'] = 'positive'
jamaal_df.loc[jamaal_df['positive_percentage'] < jamaal_df['negative_percentage'], 'predicted_sentiment_score'] = 'negative'
jamaal_df.loc[jamaal_df['positive_percentage'] == jamaal_df['negative_percentage'], 'predicted_sentiment_score'] = 'neutral'

jamaal_df

# %%
from sklearn.metrics import accuracy_score, f1_score

# Compare sentiment and predicted_sentiment_score columns
accuracy = accuracy_score(jamaal_df['sentiment'], jamaal_df['predicted_sentiment_score'])
f1 = f1_score(jamaal_df['sentiment'], jamaal_df['predicted_sentiment_score'], average='weighted')

accuracy, f1



