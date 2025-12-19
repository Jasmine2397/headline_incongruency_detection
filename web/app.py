from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

app = Flask(__name__)

# Load trained model and vectorizer
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    cosine_score = None

    if request.method == "POST":
        headline = clean_text(request.form["headline"])
        article = clean_text(request.form["article"])

        # TF-IDF vectors
        head_vec = tfidf.transform([headline])
        art_vec = tfidf.transform([article])

        # Cosine similarity
        head_norm = normalize(head_vec)
        art_norm = normalize(art_vec)
        cosine_sim = head_norm.multiply(art_norm).sum(axis=1).A
        cosine_score = round(float(cosine_sim[0][0]), 4)

        # Feature combination
        X = hstack([head_vec, art_vec, cosine_sim])

        # Prediction
        pred_class = model.predict(X)[0]
        prediction = LABEL_MAP[pred_class]

        # ---------- CONFIDENCE LOGIC ----------
        decision_scores = model.decision_function(X)

        # Convert scores to probabilities
        probs = softmax(decision_scores[0])
        confidence = round(float(probs[pred_class] * 100), 2)
        # --------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        cosine_score=cosine_score
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

