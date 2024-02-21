# %%
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np

# %%
jamaal_df = pd.read_csv('./COVID19_mini.csv')
pd.set_option('display.max_colwidth',1)
jamaal_df.head()

# %%
jamaal_df.info()

# %%
jamaal_df= jamaal_df.drop(["user"],axis=1)
jamaal_df

# %%
jamaal_df['text'] = jamaal_df['text'].str.split(':', n=1).str[1]
jamaal_df


# %%
jamaal_df['text'] = jamaal_df['text'].str.lstrip("...")
jamaal_df


# %%
jamaal_df['text'] = jamaal_df['text'].str.replace('@', '')
jamaal_df['text'] = jamaal_df['text'].str.replace('#', '')
jamaal_df

# %%
jamaal_df['text'] = jamaal_df['text'].str.lower()
jamaal_df

# %%
jamaal_df['text'] = jamaal_df['text'].str.replace("\d+", "",regex=True)
jamaal_df

# %%
import string

jamaal_df['text'] = jamaal_df['text'].str.translate(str.maketrans('', '', string.punctuation))
jamaal_df

# %%
jamaal_df['text'] = jamaal_df['text'].str.replace('…', '')
jamaal_df


# %%
jamaal_df['text'] = jamaal_df['text'].str.replace('“', '')
jamaal_df

# %%
missing_rows = jamaal_df.isnull().sum(axis=1)
num_missing_rows = len(missing_rows[missing_rows > 0])
print(f"Number of missing rows: {num_missing_rows}")

# %%
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

gn_vec_path = "GoogleNews-vectors-negative300.bin"
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=gn_vec_path,
    action="insert")

jamaal_df_after_word_augmenter = jamaal_df.copy()

for i in range(len(jamaal_df_after_word_augmenter)):
    text = jamaal_df_after_word_augmenter['text'][i]
    augmented_text = aug.augment(text)
    jamaal_df_after_word_augmenter['text'][i] = augmented_text

jamaal_df_after_word_augmenter

# %%
jamaal_df_after_word_augmenter['text'] = jamaal_df_after_word_augmenter['text'].apply(' '.join)
jamaal_df_after_word_augmenter


# %%
jamaal_df_after_word_augmenter = pd.concat(
    [jamaal_df, jamaal_df_after_word_augmenter], ignore_index=True)


jamaal_df_after_word_augmenter

# %%
import string
jamaal_df_after_word_augmenter['text'] = jamaal_df_after_word_augmenter['text'].str.lower()
jamaal_df_after_word_augmenter['text'] = jamaal_df_after_word_augmenter['text'].str.translate(str.maketrans('', '', string.punctuation))
jamaal_df_after_word_augmenter['text'] = jamaal_df_after_word_augmenter['text'].str.replace("\d+", "", regex=True)
jamaal_df_after_word_augmenter

jamaal_df_after_word_augmenter


# %%
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import re

nltk.download('stopwords')
nltk.download('punkt')
stop_words_nltk = set(stopwords.words('english'))

# %%
text_string = ' '.join(jamaal_df_after_word_augmenter['text'])
print(text_string)

# %%
tokenized_corpus_nltk = word_tokenize(text_string)

# %%
tokenized_corpus_without_stopwords = [
    i for i in tokenized_corpus_nltk if not i in stop_words_nltk]
print("Tokenized corpus without stopwords:",
      tokenized_corpus_without_stopwords)

# %%
pretrainedpath = "GoogleNews-vectors-negative300.bin"
w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True)

# %%
aug = naw.WordEmbsAug(
    model_type='word2vec', model_path=gn_vec_path,
    action="substitute", aug_max=2, aug_min=2)
new_texts = []
for i in range(len(jamaal_df_after_word_augmenter)):
    text = jamaal_df_after_word_augmenter['text'][i]
    augmented_text = aug.augment(text)
    new_texts.append(augmented_text)

jamaal_df_after_word_augmenter = jamaal_df_after_word_augmenter.assign(new_text = new_texts)
jamaal_df_after_word_augmenter

# %%
jamaal_df_after_word_augmenter['new_text'] = jamaal_df_after_word_augmenter['new_text'].apply(
    ' '.join)
jamaal_df_after_word_augmenter

# %%
jamaal_df_after_word_augmenter.to_csv('jamaal_df_after_random_insertion.txt', sep=' ', index=False)



