import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Reading the data
movies_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/movies.csv'
ratings_path = 'https://raw.githubusercontent.com/Rakshitx1/Movie-Recomendation-System/master/Dataset/ratings.csv'
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Data Preprocessing
movie_ratings = pd.merge(ratings, movies, on='movieId')
movie_ratings['avg_rating'] = movie_ratings.groupby('movieId')['rating'].transform('mean')
movie_ratings.drop(['userId', 'timestamp', 'rating'], axis=1, inplace=True)
user_movie_ratings = movie_ratings.pivot_table(index='movieId', columns='title', values='avg_rating').fillna(0)

# Finding the optimal K
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(user_movie_ratings)
    wcss.append(kmeans.inertia_)

min_silhouette = np.min(wcss)

for i in range(len(wcss)):
    if wcss[i] == min_silhouette:
        num_clusters = i + 2
        break

# Implementing K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(user_movie_ratings)
cluster_labels = kmeans.labels_
user_movie_ratings['cluster'] = cluster_labels
