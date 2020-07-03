# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:40:02 2020

@author: Bugra
"""

import pandas as pd
import numpy as np 


df1=pd.read_csv("tmdb_5000_credits.csv")
df2=pd.read_csv("tmdb_5000_movies.csv")


#iki veri setine id değerine göre birleştiriyor
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')


#Kelimenin metinde geçiş sıklığına göre vektörlerini buluyoruz
#Metinlerin benzerlik oranını tahmin ediyoruz 
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Ingilizcedeki a ve the gibi kelimleri siliyor
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#NaN yazan yerleri boş bıraktırıyor
#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#TF-IDF matricini oluşturuyor
#TF -> Terim sayısı / Toplam sayı
#IDF -> Terimin geçtiği metin sayısı / Toplam metin sayısı
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)
#4803 filmi tanımlamak için 20978 kelime kullanıldığı görünüyor



#Oluşturulan matrix ile benzerlik skoru bulunacak
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Verimiz nokta veri olduğu için
# cosine_similarities() ile linear_kernel() aynı sonucu verir
# linear_kernal() daha hızlı olduğu için
# o fonksiyonu kullanıyoruz.
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


#Index ve başlık olan bir set oluşturdu
#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# %%

# Verilen film başlığına göre en benzer 10 filmi öneren sistem
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    
    #filmi aldı 
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Seçilenin tüm filimlere benzerlik oranı alınır
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Benzerlik puanlarına göre sıralar
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # En iyi 10 taneyi alır
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Yukarda indisleri bilinen 10 filmi return ediyor
    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

print("\n\n\n")
print(get_recommendations('Avengers: Age of Ultron'))

print("\n\n\n")
print(get_recommendations('The Avengers'))

# Açıklamalardaki kelimelere göre öneri işlemi bitti 



# %% 


# Bu tahmin filimde oynayan ilk 3 aktör, yönetmen, ilgili tür ve anahtar kelimeler kullanılarak yapılacaktır
# değerleri ayrıştırmada kullanılır
# Parse the stringified features into their corresponding python objects
from ast import literal_eval


# Bu değerleri ayrıştırır
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

# Crewden yönetmenin ismini al, listede yoksa return NaN yap
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    # x liste mi diye kontrol ediyor
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    #Return empty list in case of missing/malformed data
    return []


# Özellikleri tanımlıyor
# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    
    

# Bu işlem tüm harfleri küçük harfe çeviriyor
# Kelimelerin arasındaki boşlukları kaldırıryor
# Aynı isme sahip 2 yönetmenin benzerlik oranını ortadan kaldırmak için 
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        


# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
    

# Tüm verilerin bulunduğu metadata soup'u oluşturuyoruz.
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)


# TF-IDF yerine CountVectorizer kullandık
# Bir oyuncu çok fazla filmde oynamışsa, ağırlığını dengelemek için
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

# A, the gibi kelimeleri sildik
count = CountVectorizer(stop_words='english')
# Count Vectörü oluşturduk
count_matrix = count.fit_transform(df2['soup'])


# Benzerlik oranını cosine similarity ile buluyoruz
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# Index ve başlık olan bir set oluşturdu
# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


print("\n\n\n")
print(get_recommendations('Avengers: Age of Ultron', cosine_sim2))

print("\n\n\n")
print(get_recommendations('The Godfather', cosine_sim2))















