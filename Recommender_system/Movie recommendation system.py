#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


#import the dataset
df = pd.read_csv("movie_dataset.csv")


# In[4]:


features = ['keywords','cast','genres','director']


# In[5]:


#create a function for combining the values of these columns into a single string.
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[6]:


#call this function over each row of our dataframe. 
#But, before doing that, we need to clean and preprocess the data for our use. 
#We will fill all the NaN values with blank string in the dataframe.
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string

df["combined_features"] = df.apply(combine_features,axis=1)


# In[7]:


df.iloc[0].combined_features


# In[8]:


cv = CountVectorizer() #creating new CountVectorizer() object
count_matrix = cv.fit_transform(df["combined_features"]) 
#feeding combined strings(movie contents) to CountVectorizer() object


# In[9]:


#Now, we need to obtain the cosine similarity matrix from the count matrix.
cosine_sim = cosine_similarity(count_matrix)


# In[10]:


#Now, we will define two helper functions to get movie title from movie index and vice-versa.
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[11]:


print(df["original_title"])
movie_user_likes=input('enter the movie name from the above list')
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[12]:


#We will sort the list similar_movies according to similarity scores in descending order. 
#Since the most similar movie to a given movie will be itself, we will discard the first element after sorting the movies.
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


# In[13]:


#we will run a loop to print first 5 entries from sorted_similar_movies list.
i=0
print("The top 5 similar movies to "+movie_user_likes+" are the following:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break

