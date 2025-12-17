import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

DATA_PATH = "../data/processed/chunked_dataset.csv"


def main():
    print("Training final model and saving artifacts...")

    df = pd.read_csv(DATA_PATH)

    X_head = df["headline"].astype(str)
    X_art = df["article_chunk"].astype(str)
    y = df["label"]

    # TF-IDF (shared space)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf.fit(pd.concat([X_head, X_art]))

    X_head_vec = tfidf.transform(X_head)
    X_art_vec = tfidf.transform(X_art)

    # Cosine similarity
    X_head_norm = normalize(X_head_vec)
    X_art_norm = normalize(X_art_vec)
    cosine_sim = X_head_norm.multiply(X_art_norm).sum(axis=1).A

    X = hstack([X_head_vec, X_art_vec, cosine_sim])

    # Train SVM
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, "../web/model/svm_model.joblib")
    joblib.dump(tfidf, "../web/model/tfidf.joblib")

    print("Model and TF-IDF vectorizer saved successfully.")


if __name__ == "__main__":
    main()
