import pandas as pd
import os
import re
import spacy

# Load spaCy safely for sentence segmentation
nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "tagger", "parser", "lemmatizer"]
)
nlp.add_pipe("sentencizer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "final_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "chunked_dataset.csv")

MAX_SENTENCES_PER_CHUNK = 5


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_article(article):
    doc = nlp(article)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    for i in range(0, len(sentences), MAX_SENTENCES_PER_CHUNK):
        chunk = " ".join(sentences[i:i + MAX_SENTENCES_PER_CHUNK])
        chunks.append(chunk)

    return chunks


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"Processed {i} articles...")

        headline = clean_text(row["headline"])
        article = clean_text(row["article"])
        label = row["label"]

        if not article:
            continue

        chunks = chunk_article(article)

        for chunk in chunks:
            rows.append({
                "headline": headline,
                "article_chunk": chunk,
                "label": label
            })

    processed_df = pd.DataFrame(rows)

    print("Original samples:", len(df))
    print("Chunked samples:", len(processed_df))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nChunked dataset saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
