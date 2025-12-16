import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from dataset_loader import load_dataset


LABEL_NAMES = {
    0: "congruent",
    1: "weakly_incongruent",
    2: "strongly_incongruent",
    3: "contradictory"
}


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_dataset()

    # Train final model
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Load raw text for inspection
    df = pd.read_csv("../data/processed/chunked_dataset.csv")
    _, df_test = df.iloc[:len(y_train)], df.iloc[len(y_train):]

    df_test = df_test.reset_index(drop=True)
    df_test["true_label"] = y_test.values
    df_test["pred_label"] = y_pred

    # Identify errors
    errors = df_test[df_test["true_label"] != df_test["pred_label"]]

    print(f"\nTotal test samples: {len(df_test)}")
    print(f"Total misclassified samples: {len(errors)}")

    # Save misclassified samples
    errors["true_label_name"] = errors["true_label"].map(LABEL_NAMES)
    errors["pred_label_name"] = errors["pred_label"].map(LABEL_NAMES)

    errors = errors[[
        "headline",
        "article_chunk",
        "true_label_name",
        "pred_label_name"
    ]]

    errors.to_csv("../data/processed/error_analysis.csv", index=False)

    print("\nSaved misclassified samples to:")
    print("../data/processed/error_analysis.csv")

    # Show common confusion pairs
    print("\nMost common confusions:")
    confusion_pairs = (
        errors.groupby(["true_label_name", "pred_label_name"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(confusion_pairs)


if __name__ == "__main__":
    main()
