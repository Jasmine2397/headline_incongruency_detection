import pandas as pd
import numpy as np

# File paths
STANCES_PATH = "../data/raw/train_stances.csv"
BODIES_PATH = "../data/raw/train_bodies.csv"

def main():
    # Load datasets
    stances_df = pd.read_csv(STANCES_PATH)
    bodies_df = pd.read_csv(BODIES_PATH)

    print("Stances shape:", stances_df.shape)
    print("Bodies shape:", bodies_df.shape)

    # Merge headline with article
    merged_df = stances_df.merge(
        bodies_df,
        how="left",
        on="Body ID"
    )

    print("\nMerged dataset shape:", merged_df.shape)
    print("\nSample rows:")
    print(merged_df.head(3))

    # Label distribution
    print("\nLabel distribution:")
    print(merged_df["Stance"].value_counts())

    # Article length analysis
    merged_df["article_length"] = merged_df["articleBody"].apply(
        lambda x: len(str(x).split())
    )

    print("\nArticle length statistics:")
    print(merged_df["article_length"].describe())

if __name__ == "__main__":
    main()
