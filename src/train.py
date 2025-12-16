import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset_loader import load_dataset


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("\nTraining Linear SVM...")
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "congruent",
            "weakly_incongruent",
            "strongly_incongruent",
            "contradictory"
        ]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
