import pandas as pd
import numpy as np
from .collaborative import get_svd_recommendations
from .content_based import get_content_recommendations
from .data_loader import load_books, load_ratings

def min_max_normalize(scores):
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

def get_hybrid_recommendations(user_id, n=10, cf_weight=0.5):
    books_df = load_books()
    ratings_df = load_ratings()
    
    # Check if user exists (Cold Start check)
    if user_id not in ratings_df['user_id'].values:
        return None  # Trigger Cold Start
    
    # Get CF recommendations
    cf_recs = get_svd_recommendations(user_id, books_df, ratings_df, n=n*2)
    
    # Get CB recommendations
    cb_recs = get_content_recommendations(user_id, ratings_df, books_df, n=n*2)
    
    if not cf_recs or not cb_recs:
        # Fallback to whatever is available
        return cf_recs if cf_recs else cb_recs
    
    # Create dictionaries for quick lookup
    cf_dict = {book_id: score for book_id, score in cf_recs}
    cb_dict = {book_id: score for book_id, score in cb_recs}
    
    # Collect all unique book IDs from both lists
    all_book_ids = set(cf_dict.keys()) | set(cb_dict.keys())
    
    hybrid_scores = []
    for book_id in all_book_ids:
        cf_score = cf_dict.get(book_id, 0)
        cb_score = cb_dict.get(book_id, 0)
        hybrid_scores.append((book_id, (cf_weight * cf_score) + ((1 - cf_weight) * cb_score)))
    
    # Sort and return top N
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    return hybrid_scores[:n]

# ---------- FIXED COLD START FUNCTION ----------
def get_cold_start_recommendations(genres_list, n=10):
    books_df = load_books()
    
    # Define a function to check if a book matches ANY selected genre
    def matches_genre(row):
        # If the book has no genre data, skip it
        if pd.isna(row['genres']) or row['genres'] == 'General':
            return False
        
        # Split the genres by '|' and check for matches
        book_genres = str(row['genres']).lower().split('|')
        # Check if any selected genre appears in the book's genre list
        return any(g.lower() in book_genres for g in genres_list)
    
    # Filter the books
    filtered = books_df[books_df.apply(matches_genre, axis=1)]
    
    # FALLBACK: If no books match, just recommend the highest-rated books overall
    if filtered.empty:
        print("No books matched the selected genres. Showing highest-rated books instead.")
        filtered = books_df
    
    # Sort by average rating and return top N
    top_books = filtered.sort_values('average_rating', ascending=False).head(n)
    return list(zip(top_books['book_id'].values, top_books['average_rating'].values))