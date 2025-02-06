import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

# Data Loading
movies_path = 'https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/master/Dataset/movies.csv'
ratings_path = 'https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/master/Dataset/ratings.csv'
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Data Pre Processing
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Removing Noise from Data
final_dataset.fillna(0, inplace=True)

# Finding the parameters for optimal sparsity
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

param_space = {
    'min_user_votes': range(10, 101),
    'min_movie_votes': range(30, 101)
}
n_iter = 100
best_score = float('inf')
best_params = None
param_sampler = ParameterSampler(param_space, n_iter=n_iter, random_state=42)
for params in param_sampler:
    filtered_ratings = ratings.copy()
    user_counts = filtered_ratings['userId'].value_counts()
    movie_counts = filtered_ratings['movieId'].value_counts()
    filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(user_counts[user_counts >= params['min_user_votes']].index)]
    filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(movie_counts[movie_counts >= params['min_movie_votes']].index)]
    sparsity = 1 - len(filtered_ratings) / (len(ratings) * len(movies))
    if sparsity < best_score:
        best_score = sparsity
        best_params = params

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 15].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 31].index]

# Finding whether a movie is present or not
def get_movie_id(movie_name, movies_df):
    movie = movies_df[movies_df['title'].str.contains(movie_name, case=False)]
    if not movie.empty:
        return movie.iloc[0]['movieId']
    else:
        return None

# Calculating the Similarity matrix
def cosine_similarity_matrix(data):
    data_normalized = data.div(np.sqrt(np.square(data).sum(axis=1)), axis=0)
    similarity_matrix = data_normalized.dot(data_normalized.T)
    return similarity_matrix.values

cosine_sim = cosine_similarity_matrix(final_dataset)

# Implementing KNN
def find_k_nearest_neighbors(movie_idx, k=20):
    distances = cosine_sim[movie_idx]
    nearest_neighbors = np.argsort(distances)[-(k+1):]
    nearest_distances = distances[nearest_neighbors]
    return nearest_neighbors, nearest_distances
