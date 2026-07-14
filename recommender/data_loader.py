import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/')

def load_books():
    books_df = pd.read_csv(os.path.join(DATA_PATH, 'books.csv'))
    
    # Create description for content-based filtering
    if 'description' not in books_df.columns:
        books_df['description'] = books_df['title'] + " " + books_df['authors'].fillna('')
    else:
        books_df['description'] = books_df['description'].fillna(books_df['title'])

    # ----------------------------------------------
    # NEW: Load genres from the tag files
    # ----------------------------------------------
    try:
        # Load the tag mapping files
        book_tags = pd.read_csv(os.path.join(DATA_PATH, 'book_tags.csv'))
        tags = pd.read_csv(os.path.join(DATA_PATH, 'tags.csv'))
        
        # Merge to get the actual tag names (e.g., "Fantasy", "Romance")
        book_tags = book_tags.merge(tags, on='tag_id', how='left')
        
        # Take the top 5 most popular tags for each book
        top_tags = book_tags.sort_values('count', ascending=False).groupby('goodreads_book_id').head(5)
        
        # Combine them into a single string separated by '|' (e.g., "Fantasy|Romance|Young Adult")
        genres_series = top_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: '|'.join(x))
        
        # Map these genres to our main books_df using 'book_id'
        # (In this dataset, 'goodreads_book_id' matches 'book_id')
        books_df['genres'] = books_df['book_id'].map(genres_series)
        
        # If a book has no tags, label it as 'General'
        books_df['genres'] = books_df['genres'].fillna('General')
        
        print(f"✅ Loaded {len(books_df)} books with genres successfully!")
        
    except FileNotFoundError:
        print("⚠️ Warning: book_tags.csv or tags.csv not found. Cold Start will use fallback.")
        books_df['genres'] = 'General'
    except Exception as e:
        print(f"⚠️ Error loading genres: {e}. Using fallback.")
        books_df['genres'] = 'General'

    return books_df

def load_ratings():
    df = pd.read_csv(os.path.join(DATA_PATH, 'ratings.csv'))
    return df

def get_book_by_id(book_id, books_df):
    return books_df[books_df['book_id'] == book_id].iloc[0] if not books_df[books_df['book_id'] == book_id].empty else None

def get_books_by_ids(ids, books_df):
    return books_df[books_df['book_id'].isin(ids)]