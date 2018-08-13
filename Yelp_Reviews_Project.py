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

sns.set_style('darkgrid')

yelp = pd.read_csv('yelp.csv')											# Add text length as a column in dataframe

# This dataset has 10000 entries, columns are business_id, date, review_id, stars, text, type, user_id, cool, useful, funny
# Columns stars, cool, useful, funny are integers. The last 3 are counts for a readers reaction to a written review. 


### Data Analysis

yelp['text_length'] = yelp['text'].apply(len)

fig_stars = sns.FacetGrid(yelp,col='stars')
fig_stars = fig_stars.map(plt.hist, 'text_length')						# bar graph of text length for each star ranking


fig_box = plt.figure()
sns.boxplot(x='stars', y='text_length', data=yelp, palette='plasma')	# boxplot representation of each star rating. 
																		# Indicates where medians and quartiles are located for each star rating

fig_count = plt.figure()
sns.countplot(x='stars', data=yelp, palette='plasma')					# counts the number of occurences of each star rating

fig_heat = plt.figure()	
stars = yelp.groupby('stars').mean()									# group the data set by star ratings using the average value for each star rating.
sns.heatmap(stars.corr(), cmap='RdBu_r', annot=True)					# plot a heatmap for the correlation between review traits. 


plt.show()

### Notes: 

# Useful, Funny, and Text Length have relatively high coorelation while, the Cool rating has negative coorelation with all the traits. 
# 1 star reviews tend to have more text, and readers rate these reviews as funny, or useful more often. 
# 5 star reviews tend to have more ratings labeled as cool in comparison to 1 star reviews. 
# However its worth noting the average cool rating for a 5 star review is less than 1. Still not very frequent. 
# Overall, 1 star reviews get more attention than 5 star reviews from both the reviewer and reader perspective. 
