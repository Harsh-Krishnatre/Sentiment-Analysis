from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
import pandas as pd

vectorizer = CountVectorizer()

def preprocess_data(df):
    x = vectorizer.fit_transform(df["text"])
    df['label'] = df['label'].astype(int)  # FIXED
    y = df["label"]
    return x, y

def preprocess_test_data(df):
    x = vectorizer.transform(df["text"])
    df['label'] = df['label'].astype(int)  # FIXED
    y = df["label"]
    return x, y
