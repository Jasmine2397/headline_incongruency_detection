import pandas as pd

INPUT_PATH = "../data/raw/train_stances.csv"
OUTPUT_PATH = "../data/processed/mapped_stances.csv"

LABEL_MAP = {
    "agree": "congruent",
    "discuss": "weakly_incongruent",
    "unrelated": "strongly_incongruent",
    "disagree": "contradictory"
}

def main():
    df = pd.read_csv(INPUT_PATH)

    # Map labels
    df["incongruency_label"] = df["Stance"].map(LABEL_MAP)

    # Check mapping
    print("Mapped label distribution:")
    print(df["incongruency_label"].value_counts())

    # Save mapped data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMapped dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
