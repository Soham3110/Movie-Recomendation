import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data Loading
rating_path = "https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/main/Dataset/ratings.csv"
movies_path = 'https://raw.githubusercontent.com/Bansal0527/Movie-Recomendation-System/master/Dataset/movies.csv'
ratings = pd.read_csv(rating_path)
movies = pd.read_csv(movies_path)

# Define threshold for liked/disliked
threshold = 2

# Convert ratings to binary labels
ratings['liked'] = (ratings['rating'] >= threshold).astype(int)

# Prepare data for logistic regression
X = ratings[['userId', 'movieId']]
y = ratings['liked']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
