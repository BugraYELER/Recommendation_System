# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:52:43 2020

@author: Bugra
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# User which is film recomodated 
user_movie_id = [1,3,6,47,50,70,101,110]
user_rating = [4,4,4,5,5,3,5,4]
user_id = [100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000]


my_rating = pd.DataFrame({'userId' : user_id, 'movieId': user_movie_id,'rating' : user_rating})


new_rating = pd.concat([ratings,my_rating],axis=0,ignore_index=True)

# User film rating matrices
matrices = new_rating.pivot(index = "userId",columns = "movieId",values = "rating")


def standardize(row):
    return (row - row.mean())/(row.max()-row.min())



user_movie_ratings = matrices[user_movie_id]
user_movie_ratings = user_movie_ratings.fillna(0)
user_movie_ratings = user_movie_ratings.apply(standardize)


# Use cosine similerity for user-user similarity
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity = pd.DataFrame(user_similarity,columns = matrices.index,index = matrices.index)



my_row = user_similarity.iloc[-1,:]
my_row = my_row.sort_values(ascending = False)
most_similar = my_row.iloc[1:6]
most_similar_user = most_similar.index.to_list()
most_similar_score = list(my_row.iloc[1:6].values)



not_watched = matrices.drop(columns = user_movie_id)
not_watched = not_watched.T[most_similar_user]
not_watched = not_watched[not_watched.isnull().sum(axis=1) < 3]


my_dict=dict()

for i in range(len(most_similar_user)):
    my_dict[most_similar_user[i]] = most_similar_score[i]
    
for i in most_similar_user:
    not_watched[i]*=my_dict[i]
    
score = []


for i in range(not_watched.shape[0]):
    s1 = 0
    s2 = 0
    for j in range(not_watched.shape[1]):
        if(np.isnan(not_watched.iloc[i,j])):
            continue
        else:
            s1 = s1+not_watched.iloc[i,j]
            s2 = s2 + my_dict[not_watched.columns[j]]
            
    s1 = s1/s2
    score.append(s1)
not_watched['score'] = score
not_watched = not_watched.sort_values(by = "score",ascending = False)



movie_list = not_watched.iloc[:10,-1]
output = movies.loc[movies['movieId'].isin(movie_list.index.to_list())]
print(output["title"])