#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install numpy pandas scikit-learn')



# # Text analysing and creating dataframe and making a suitable dataset

# In[7]:


# importing modules

import pandas as pd
import numpy as np
import sklearn


# In[8]:


# importing the datasets
credit_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')
movies_df


# In[9]:


credit_df


# In[10]:


# Marging the two datasets based on commomn column 'title'
movies_df = movies_df.merge(credit_df, on='title')


# In[11]:


# data information
movies_df.info()
movies_df.shape


# In[12]:


# taking the main working columns 
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies_df.head(5)


# In[13]:


# after changing columns the information will be
movies_df.info()
movies_df.shape


# In[14]:


# removing null values 
movies_df.isnull().sum()


# In[15]:


movies_df.dropna(inplace=True)
movies_df.isnull().sum()


# In[16]:


# checking for duplicated values
movies_df.duplicated().sum()


# In[17]:


movies_df


# In[18]:


movies_df.iloc[0].genres


# In[19]:


# converting like normal one using ast module
import ast

# just adding the names only in genres and keywords column
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[20]:


movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df


# In[21]:


# just adding only 3 actors/actress in cast column
def convert_for_cast(obj):
    l = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            l.append(i['name'])
            count += 1
        else:
            break
    return l


# In[22]:


movies_df['cast'] = movies_df['cast'].apply(convert_for_cast)


# In[23]:


movies_df


# In[24]:


movies_df.iloc[0].crew
# selecting only the director

def convert_crew(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            l.append(i['name'])
    return l

movies_df['crew'] = movies_df['crew'].apply(convert_crew)


# In[25]:


movies_df


# In[26]:


movies_df['overview'][0]


# In[27]:


# make the overview compatible because it is too large to recommend user by these long string

movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())


# In[28]:


movies_df


# In[29]:


# we want to remove the extraspaces from the below columns 
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[30]:


movies_df


# In[31]:


# now we have to merge all the data columns in a single column
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['cast'] + movies_df['crew'] + movies_df['keywords']


# In[32]:


movies_df


# In[33]:


# now we are going to make a new dataframe include movie_id, title and tag column

new_df = movies_df[['movie_id', 'title', 'tags']]


# In[34]:


new_df


# In[35]:


# now we are removing the list format from the tags column and convert it to a string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df


# In[36]:


new_df.iloc[0].tags


# In[37]:


# making it lower letter for better prediction compatibility
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'][0]


# #  Prediction Started

# In[38]:


# we are now make some feature extraction from the new_df
# we are using CountVectorizer which is a useful tool that converts texts to vectors on the basis of the 
# frequency count of each word that occur in the entire text


# In[61]:


from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer to convert text data from the 'tags' column in your dataframe into a numerical feature 
#representation (count of words/tokens). The CountVectorizer is a useful tool for text processing and is 
#commonly used in natural language processing tasks.
cv = CountVectorizer(max_features=len(new_df), stop_words='english')


# In[63]:


vector = cv.fit_transform(new_df['tags']).toarray()
# converting the vector to array


# In[66]:


len(cv.get_feature_names_out())


# In[67]:


import nltk
#NLTK (Natural Language Toolkit) is a Python package for working with human language data, providing tools for 
#text processing, linguistic analyses, and language modeling. It is widely used in natural language processing 
#tasks and research.


# In[70]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

'''The PorterStemmer is a widely used stemming algorithm for the English language. Stemming is the process 
of reducing inflected words to their root form (stem) by removing suffixes and prefixes. It helps in simplifying 
the words and reducing them to a common base, which is useful for various natural language processing tasks like 
text classification, information retrieval, and clustering.'''


# In[74]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
'''stem(text) is a custom function that takes a string of text as input and applies stemming using 
 the PorterStemmer to each word in the text. It then returns the stemmed text as a single string.'''


# In[78]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[79]:


from sklearn.metrics.pairwise import cosine_similarity

'''cosine_similarity function from scikit-learn's sklearn.metrics.pairwise module. The cosine_similarity function 
calculates the cosine similarity between vectors in a given matrix. Cosine similarity is a measure of similarity 
between two non-zero vectors in an inner product space and is often used for comparing the similarity between two 
documents or text data represented as vectors.'''

cosine_similarity(vector)


# In[83]:


cosine_similarity(vector).shape

similarities = cosine_similarity(vector)


# In[84]:


similarities[0]


# In[85]:


similarities[0].shape


# In[87]:


sorted(list(enumerate(similarities[0])), reverse=True, key=lambda x:x[1])[1:6]
# These are some similer vectors that present and represented in reverse sorted order to only 6 th position


# # Movie recommendation function

# In[88]:


def Recommend_Movies(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarities[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:10]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[ ]:


# Function working

'''The function Recommend_Movies(movie) you provided appears to be a movie recommendation function. It takes a movie
title as input and recommends similar movies based on cosine similarity scores.

Here's a breakdown of the function:

movie_index = new_df[new_df['title'] == movie].index[0]: This line finds the index of the movie in the DataFrame 
new_df that matches the input movie title.

distances = similarities[movie_index]: This line retrieves the row corresponding to the movie_index from the 
similarities matrix. It assumes that similarities is a matrix containing cosine similarity scores between different 
movies. Each row in the similarities matrix represents the cosine similarity scores of a specific movie with all 
other movies.

movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:10]: This line sorts the cosine 
similarity scores in descending order along with their corresponding movie indices. It then selects the top 10 
similar movies (excluding the first one, which would be the input movie itself) based on the highest cosine 
similarity scores.

for i in movie_list: ...: This loop iterates through the top 10 similar movie indices in movie_list.

print(new_df.iloc[i[0]].title): For each similar movie index i[0], this line prints the title of the recommended 
movie from the DataFrame new_df.'''


# In[89]:


Recommend_Movies('Avatar')


# In[90]:


Recommend_Movies('Thor')


# In[96]:


movies_df.head(10)


# In[95]:


Recommend_Movies('The Dark Knight Rises')


# In[98]:


Recommend_Movies('Spider-Man 3')

