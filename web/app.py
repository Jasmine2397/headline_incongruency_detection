from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model/svm_model.joblib")
tfidf = joblib.load("model/tfidf.joblib")

LABEL_MAP = {
    0: "Congruent",
    1: "Weakly Incongruent",
    2: "Strongly Incongruent",
    3: "Contradictory"
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        headline = clean_text(request.form["headline"])
        article = clean_text(request.form["article"])

        # Vectorize
        head_vec = tfidf.transform([headline])
        art_vec = tfidf.transform([article])

        # Cosine similarity
        head_norm = normalize(head_vec)
        art_norm = normalize(art_vec)
        cosine_sim = head_norm.multiply(art_norm).sum(axis=1).A

        # Combine features
        X = hstack([head_vec, art_vec, cosine_sim])

        pred = model.predict(X)[0]
        prediction = LABEL_MAP[pred]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
