# Headline–Article Incongruency Detection

This project detects semantic incongruency between news headlines and their corresponding article content using classical NLP and machine learning techniques.

## Dataset
- Fake News Challenge (FNC-1)

## Approach
- Text preprocessing and sentence chunking
- TF-IDF vectorization
- Cosine similarity feature between headline and article chunks
- Linear SVM classifier

## Results
- Accuracy: ~93%
- Macro F1-score: ~0.83

## Tech Stack
- Python
- scikit-learn
- spaCy
- Flask (planned for deployment)

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
