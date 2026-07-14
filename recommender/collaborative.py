import pickle
import os
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from .data_loader import load_ratings

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/')

def train_svd_model():
    print("Loading ratings for Collaborative Filtering...")
    df = load_ratings()
    
    # Surprise requires a specific format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
    
    print("Training SVD Model... (This takes 1-2 minutes)")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.4)
    algo.fit(trainset)
    
    # Evaluate
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"SVD Model trained with RMSE: {rmse}")
    
    # Save model
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(os.path.join(MODEL_PATH, 'svd_model.pkl'), 'wb') as f:
        pickle.dump(algo, f)
    print("SVD Model saved.")
    return algo

def load_svd_model():
    try:
        with open(os.path.join(MODEL_PATH, 'svd_model.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return train_svd_model()

def get_svd_recommendations(user_id, books_df, ratings_df, n=10):
    svd_model = load_svd_model()
    # Get books the user hasn't rated
    rated_books = ratings_df[ratings_df['user_id'] == user_id]['book_id'].values
    all_books = books_df['book_id'].values
    unrated = [b for b in all_books if b not in rated_books]
    
    predictions = []
    for book_id in unrated:
        pred = svd_model.predict(user_id, book_id)
        predictions.append((book_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]