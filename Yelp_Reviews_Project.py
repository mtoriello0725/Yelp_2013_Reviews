"""
Created on Fri Aug 10 08:45:36 2018

@author: mtoriello0725
"""

"""

Python for Data Science and Machine Learning 

Yelp Reviews

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt     # plt.show() to display plot
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

sns.set_style('darkgrid')

yelp = pd.read_csv('yelp.csv')

# This dataset has 10000 entries, columns are business_id, date, review_id, stars, text, type, user_id, cool, useful, funny
# Columns stars, cool, useful, funny are integers. The last 3 are counts for a readers reaction to a written review. 


### Data Analysis

yelp['text_length'] = yelp['text'].apply(len)

g = sns.facetgrid(yelp,col='stars')
g = g.map(plt.hist, 'text_length')										# bar graph of text length for each star ranking

sns.boxplot(x='stars', y='text_length', data=yelp, palette='plasma')	# boxplot representation of each star rating. 
																		# Indicates where medians and quartiles are located for each star rating

sns.countplot(x='stars', data=yelp, palette='plasma')					# counts the number of occurences of each star rating
	
stars = yelp.groupby('stars').mean()									# group the data set by star ratings using the average value for each star rating.
sns.heatmap(stars.corr(), cmap='RdBu_r', annot=True)					# plot a heatmap for the correlation between review traits. 

# Useful, Funny, and Text Length have relatively high coorelation while, the Cool rating has negative coorelation with all the traits. 
# 1 star reviews tend to have more text, and readers rate these reviews as funny, or useful more often. 
# 5 star reviews tend to have more ratings labeled as cool in comparison to 1 star reviews. 
# However its worth noting the average cool rating for a 5 star review is less than 1. Still not very frequent. 
# Overall, 1 star reviews get more attention than 5 star reviews from both the reviewer and reader perspective. 


### NLP Classification

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]					# create new dataframe of just 1 and 5 star reviews.

Xtext, ystars = yelp_class['text'], yelp_class['stars']					# Define features as Xtext, and target as ystars. 

cv = CountVectorizer()													# Create a count vectorizer object
Xtext = cv.fit_transform(X)												# transforms Xtext into an array that counts keywords.











