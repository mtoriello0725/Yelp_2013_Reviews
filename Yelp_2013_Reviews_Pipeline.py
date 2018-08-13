"""
Created on Fri Aug 10 08:45:36 2018

@author: mtoriello0725
"""

"""

Python for Data Science and Machine Learning 

Yelp Reviews Text Processing using sklearn Pipeline 

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


yelp = pd.read_csv('yelp.csv')
yelp['text_length'] = yelp['text'].apply(len)							# Add text length as a column in dataframe


### Text Processing using Pipeline

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]					# create new dataframe of just 1 and 5 star reviews.

pipeline = Pipeline([ \
	('bow', CountVectorizer()), \
	('tfidf', TfidfTransformer()), \
	('classifier', MultinomialNB()) \
	])

Xtext, ystars = yelp_class['text'], yelp_class['stars']					# Define features as Xtext, and target as ystars. 

Xtext_train, Xtext_test, ystars_train, ystars_test =\
	train_test_split(Xtext, ystars, test_size=0.30)

pipeline.fit(Xtext_train, ystars_train)

pipe_Predict = pipeline.predict(Xtext_test)

print(confusion_matrix(ystars_test, pipe_Predict))
print('\n\n')
print(classification_report(ystars_test, pipe_Predict))

