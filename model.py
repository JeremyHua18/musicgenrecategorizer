import thecypher as cy
import urllib.request as urllib2
from bs4 import BeautifulSoup
import pandas as pd
import re
from unidecode import unidecode
from urllib.request import urlopen
import requests
import time
import lyricsgenius
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import csv
import matplotlib.pyplot as plt
import random
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances


from sklearn.utils import shuffle
def train():
    df_wordCount = pd.read_csv("data.csv")
    df_wordCount.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=None,
        inplace=True
    )
    genres = ['pop','death-metal','rock','r&b/soul','hip-hop/rap','alternative','metal','punk','pop/rock',]
    N = 200 # number of records to pull from each genre
    RANDOM_SEED = 200 # random seed to make results repeatable
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for g in genres: # loop over each genre
        subset = df_wordCount[df_wordCount.genre == g] # create subset
        train_set = subset.sample(n=N, random_state=RANDOM_SEED)
        test_set = subset.drop(train_set.index)
        train_df = train_df.append(train_set) # append subsets to the master sets
        test_df = test_df.append(test_set)
        train_df = shuffle(train_df)
        test_df = shuffle(test_df)

    wordCount_clf = Pipeline(
        [('vect', CountVectorizer()),
        ('clf', MultinomialNB(alpha=0.1))])

    # train our model on training data
    wordCount_clf.fit(train_df.lyrics, train_df.genre)  

    # score our model on testing data
    predicted = wordCount_clf.predict(test_df.lyrics)
    print(np.mean(predicted == test_df.genre))
   
    print(wordCount_clf.predict(
        [
            "i stand for the red white and blue",
            "flow so smooth they say i rap in cursive", #bars *insert fire emoji*
            "take my heart and carve it out",
            "there is no end to the madness",
            "sitting on my front porch drinking sweet tea",
            "sitting on my front porch sippin on cognac",
            "dog died and my pick up truck wont start",
            "im invisible and the drugs wont help",
            "i love you",
            "i wonder what genre a song about data science and naive bayes and hyper parameters and maybe a little scatter plots would be"
        ]
    ))

val = input("Enter your value: ") 
print(val)
train()


