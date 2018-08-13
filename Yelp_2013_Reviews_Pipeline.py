"""
Created on Fri Aug 10 08:45:36 2018

@author: mtoriello0725
"""

"""

Python for Data Science and Machine Learning 

Yelp Reviews: Text Processing using sklearn TfidfTransformer 

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
	])																	# bow - bag of words. strings to token integer counts.
																		# tfidf - term frequency-inverse document frequency
																		# integer counts to weighted TF-IDF scores. Lowers weight for commonly used words.
																		# Train dataset on the MultiNomialNB classifier. 


Xtext, ystars = yelp_class['text'], yelp_class['stars']					# Define features as Xtext, and target as ystars. 

Xtext_train, Xtext_test, ystars_train, ystars_test =\
	train_test_split(Xtext, ystars, test_size=0.30)						# Split data into training and test datasets.

pipeline.fit(Xtext_train, ystars_train)									# Fit the training data to pipeline

pipe_Predict = pipeline.predict(Xtext_test)								# predict star rating

print(confusion_matrix(ystars_test, pipe_Predict))
print('\n\n')
print(classification_report(ystars_test, pipe_Predict))					# print the confusion matrix and classification report. 

'''

Using the TF-IDF Transformer turned out to have worse results. 
The model was unable to make significant predictions for 1 star ratings. 
Almost all predictions were classified as 5 star ratings, which is not true.


'''

