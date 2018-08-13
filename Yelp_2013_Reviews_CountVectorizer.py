"""
Created on Fri Aug 10 08:45:36 2018

@author: mtoriello0725
"""

"""

Python for Data Science and Machine Learning 

Yelp Reviews: Natural Language Processing using sklearn CountVectorizer

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

yelp = pd.read_csv('yelp.csv')
yelp['text_length'] = yelp['text'].apply(len)							# Add text length as a column in dataframe

### NLP Classification

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]					# create new dataframe of just 1 and 5 star reviews.

Xtext, ystars = yelp_class['text'], yelp_class['stars']					# Define features as Xtext, and target as ystars. 

cv = CountVectorizer()													# Create a count vectorizer object
Xtext = cv.fit_transform(Xtext)											# transforms Xtext into an array that counts keywords 
																		# and weighs frequently used words accordingly.


### Model Training

Xtext_train, Xtext_test, ystars_train, ystars_test =\
	train_test_split(Xtext, ystars, test_size=0.30)

nb = MultinomialNB()
nb.fit(Xtext_train, ystars_train)

nbPredict = nb.predict(Xtext_test)

print(confusion_matrix(ystars_test, nbPredict))
print('\n\n')
print(classification_report(ystars_test, nbPredict))

'''

There is a much larger sample size for 5 star reviews than 1 star reviews. 
Results show relatively low percision for 1 star review predictions at around 65%. 
This is most likely because of the inbalance of data. 
f1-score turns out to be 74% for 1 star predictions, and 95% for 5 star predictions. 


'''