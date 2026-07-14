import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import load_books

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/')

def build_content_matrix():
    print("Loading books for Content-Based Filtering...")
    books_df = load_books()
    
    # Use description for TF-IDF. If none, use title+authors
    texts = books_df['description'].fillna(books_df['title'] + " " + books_df['authors'].fillna(''))
    
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(texts)
    
    print("Computing Cosine Similarity matrix... (This uses memory)")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save matrix
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(os.path.join(MODEL_PATH, 'cosine_sim.pkl'), 'wb') as f:
        pickle.dump(cosine_sim, f)
    print("Cosine Similarity matrix saved.")
    return cosine_sim, books_df

def load_content_matrix():
    try:
        with open(os.path.join(MODEL_PATH, 'cosine_sim.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return build_content_matrix()[0]

def get_content_recommendations(user_id, ratings_df, books_df, n=10):
    # Find the user's top 3 highest rated books
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        return None
    
    top_books = user_ratings.sort_values('rating', ascending=False).head(3)['book_id'].values
    cosine_sim = load_content_matrix()
    
    # Get indices of these books
    book_indices = {}
    for idx, row in books_df.iterrows():
        book_indices[row['book_id']] = idx
    
    # Average similarity scores of top 3 books
    avg_scores = np.zeros(len(books_df))
    for book_id in top_books:
        if book_id in book_indices:
            idx = book_indices[book_id]
            avg_scores += cosine_sim[idx]
    avg_scores = avg_scores / len(top_books)
    
    # Get top N (excluding books user already rated)
    rated_set = set(user_ratings['book_id'].values)
    sim_scores = list(enumerate(avg_scores))
    sim_scores = [(books_df.iloc[i]['book_id'], score) for i, score in sim_scores if books_df.iloc[i]['book_id'] not in rated_set]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    return sim_scores[:n]