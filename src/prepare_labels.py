import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "merged_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "final_dataset.csv")

LABEL_MAP = {
    "agree": 0,        # Congruent
    "discuss": 1,      # Weakly incongruent
    "unrelated": 2,    # Strongly incongruent
    "disagree": 3      # Contradictory
}

def main():
    df = pd.read_csv(INPUT_PATH)

    df["label"] = df["Stance"].map(LABEL_MAP)

    print("Final label distribution:")
    print(df["label"].value_counts().sort_index())

    df = df.rename(columns={
        "Headline": "headline",
        "articleBody": "article"
    })

    df = df[["headline", "article", "label"]]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFinal dataset saved to:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
