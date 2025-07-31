from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def preprocess_test_data(df, vectorizer):
    df["review"] = df["review"].apply(clean_text)
    x = vectorizer.transform(df["review"])
    y = df["sentiment"]
    return x, y

def preprocess_data(df, vectorizer=None):
    df["review"] = df["review"].apply(clean_text)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),max_features=5000)
        x = vectorizer.fit_transform(df["review"])
    else:
        x = vectorizer.transform(df["review"])
    y = df["sentiment"]
    return x, y, vectorizer