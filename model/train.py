import os
import pandas as pd
import joblib
from utils.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

MODEL_PATH = "model.pkl"

def train_model():
    path = input("Path to training CSV: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return

    df = pd.read_csv(path).dropna()
    x, y = preprocess_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)
    if joblib.dump(model, MODEL_PATH):
        print("Model saved as model.pkl")

    y_pred = model.predict(x_test)
    print("\n=== Training Report ===")
    print(classification_report(y_test, y_pred))


