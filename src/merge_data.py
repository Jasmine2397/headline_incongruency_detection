import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STANCE_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train_stances.csv")
BODY_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train_bodies.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "merged_data.csv")

def main():
    stances = pd.read_csv(STANCE_PATH)
    bodies = pd.read_csv(BODY_PATH)

    merged = stances.merge(
        bodies,
        how="left",
        left_on="Body ID",
        right_on="Body ID"
    )

    print("Merged dataset shape:", merged.shape)
    print("\nLabel distribution:")
    print(merged["Stance"].value_counts())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nMerged dataset saved at:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
