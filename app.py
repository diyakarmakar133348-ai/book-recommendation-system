from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from recommender.hybrid import get_hybrid_recommendations, get_cold_start_recommendations
from recommender.data_loader import load_books

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this for production

books_df = load_books()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = get_hybrid_recommendations(user_id, n=12)
    
    if recommendations is None:
        # Trigger Cold Start: save user_id in session and redirect to genre picker
        session['new_user_id'] = user_id
        return redirect(url_for('cold_start'))
    
    # Fetch book details for display
    book_ids = [book_id for book_id, score in recommendations]
    recommended_books = books_df[books_df['book_id'].isin(book_ids)]
    
    # Add scores to the dataframe for display
    score_dict = {book_id: round(score, 2) for book_id, score in recommendations}
    recommended_books['hybrid_score'] = recommended_books['book_id'].map(score_dict)
    recommended_books = recommended_books.sort_values('hybrid_score', ascending=False)
    
    return render_template('recommend.html', books=recommended_books.to_dict('records'), user_id=user_id)

@app.route('/cold_start', methods=['GET'])
def cold_start():
    return render_template('cold_start.html')

@app.route('/cold_start_recommend', methods=['POST'])
def cold_start_recommend():
    genres = request.form.getlist('genres')
    if not genres:
        return "Please select at least one genre.", 400
    
    recs = get_cold_start_recommendations(genres, n=12)
    book_ids = [book_id for book_id, score in recs]
    recommended_books = books_df[books_df['book_id'].isin(book_ids)]
    
    score_dict = {book_id: round(score, 2) for book_id, score in recs}
    recommended_books['hybrid_score'] = recommended_books['book_id'].map(score_dict)
    recommended_books = recommended_books.sort_values('hybrid_score', ascending=False)
    
    return render_template('recommend.html', books=recommended_books.to_dict('records'), user_id="New User")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)