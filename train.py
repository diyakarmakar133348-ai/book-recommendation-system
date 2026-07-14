from recommender.collaborative import train_svd_model
from recommender.content_based import build_content_matrix

if __name__ == "__main__":
    print("Building Content Matrix...")
    build_content_matrix()
    print("Training SVD...")
    train_svd_model()
    print("All models trained successfully!")