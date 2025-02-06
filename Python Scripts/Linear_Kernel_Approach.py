import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterSampler

# Define the preprocess_data function
def preprocess_data(ratings, min_user_votes=15, min_movie_votes=31):
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= min_user_votes].index)]
    ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_votes].index)]

    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    return final_dataset

# Loading the Data
def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

movies_path = 'https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/master/Dataset/movies.csv'
ratings_path = 'https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/master/Dataset/ratings.csv'
movies, ratings = load_data(movies_path, ratings_path)

# Preprocessing the Data
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

final_dataset = preprocess_data(ratings)

# Training the Model (Linear Kernel SVM)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

svm_model = SVC(kernel='linear')
svm_model.fit(tfidf_matrix, movies['title'])
