import os
import pandas as pd
import joblib
from utils.preprocess import preprocess_test_data
from sklearn.metrics import classification_report

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def test_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model/vectorizer not found. Train the model first.")
        return

    path = input("Path to test CSV: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return

    df = pd.read_csv(path).dropna()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    x_test, y_test = preprocess_test_data(df, vectorizer)
    y_pred = model.predict(x_test)

    print("\n=== Test Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    df["Predicted"] = y_pred
    print("\n=== Sample Predictions ===")
    print(df.head(10))
