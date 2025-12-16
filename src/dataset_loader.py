import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import hstack


DATA_PATH = "../data/processed/chunked_dataset.csv"


def load_dataset(test_size=0.2, random_state=42):
    print("Loading dataset with cosine similarity features...")

    # Load data
    df = pd.read_csv(DATA_PATH)

    X_head = df["headline"].astype(str)
    X_art = df["article_chunk"].astype(str)
    y = df["label"]

    # Train-test split
    X_head_train, X_head_test, X_art_train, X_art_test, y_train, y_test = train_test_split(
        X_head, X_art, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Shared TF-IDF vectorizer (IMPORTANT)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    # Fit on combined text to ensure same feature space
    tfidf.fit(pd.concat([X_head_train, X_art_train]))

    # Transform text
    X_head_train_vec = tfidf.transform(X_head_train)
    X_art_train_vec = tfidf.transform(X_art_train)

    X_head_test_vec = tfidf.transform(X_head_test)
    X_art_test_vec = tfidf.transform(X_art_test)

    # Normalize vectors for cosine similarity
    X_head_train_norm = normalize(X_head_train_vec)
    X_art_train_norm = normalize(X_art_train_vec)

    X_head_test_norm = normalize(X_head_test_vec)
    X_art_test_norm = normalize(X_art_test_vec)

    # Row-wise cosine similarity (memory safe)
    train_cosine = (
        X_head_train_norm.multiply(X_art_train_norm)
        .sum(axis=1)
        .A
    )

    test_cosine = (
        X_head_test_norm.multiply(X_art_test_norm)
        .sum(axis=1)
        .A
    )

    # Combine features
    X_train = hstack([
        X_head_train_vec,
        X_art_train_vec,
        train_cosine
    ])

    X_test = hstack([
        X_head_test_vec,
        X_art_test_vec,
        test_cosine
    ])

    print("Dataset loaded successfully.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test
