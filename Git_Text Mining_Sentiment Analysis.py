#!/usr/bin/env python
# coding: utf-8

# # Text Mining [Sentiment Analysis]

.............................................................................................

# ###  Elon-Musk Tweets Dataset

.............................................................................................

# Importing Libraries
import numpy as np 
import pandas as pd 
import string 
import spacy
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt


# Read the file
sent = pd.read_csv('Elon_musk.csv')
sent.head()

sent.drop(sent.columns[0], axis=1, inplace= True)
sent.head()


# To remove the leading & trailing characters
sent = [Text.strip() for Text in sent.Text]

# To remove empty strings
sent = [Text for Text in sent if Text]

# Tokenization
from nltk import tokenize
lines = tokenize.sent_tokenize(" ".join(sent))
lines[:10]


# Creating a dataframe
lines_df = pd.DataFrame(lines, columns=['Sentences'])
lines_df


# Sentiment Analysis Scores
scores = pd.read_csv('Afinn_scores.csv')
scores.head()

affinity_scores = scores.set_index('word')['value'].to_dict() 


# Using 'affinity_scores' as Sentiment Lexicon & calculating sentiment score for entire original sentence
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score 

# test score
calculate_sentiment(text = 'great') 

# Creating a new column for showing Sentiment scores for individual sentences
lines_df['Sentiment_score'] = lines_df['Sentences'].apply(calculate_sentiment)

# Creating a column for showing count of words in the sentence
lines_df['word_count'] = lines_df['Sentences'].str.split().apply(len)
lines_df['word_count'].head(10) 

# Table with 'Sentiment scores' & 'Count of words'
lines_df.tail(15)

# Top 15 sentences in terms of the highest sentiment scores
lines_df.sort_values(by='Sentiment_score', ascending= False).head(15) 

lines_df['Sentiment_score'].describe()

# Sentiment scores of zero or less than zer
lines_df[lines_df['Sentiment_score']<=0].head(10) 

# Plotting the results
# Distiribution Plot

lines_df['index'] = range(0, len(lines_df))

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize= (8, 6))
sns.distplot(lines_df['Sentiment_score'])


# Lineplot
plt.figure(figsize=(18, 10))
sns.lineplot(y='Sentiment_score', x='index', data=lines_df) 


# Scatter Plot
plt.figure(figsize= (12,12))
plt.scatter(x = 'word_count', y = 'Sentiment_score', data= lines_df)
plt.title('Word Conut vs Sentiment Score Plot')
plt.xlabel('Word Count')
plt.ylabel('Sentiment Score')
plt.show()

