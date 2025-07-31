def stringify(n):
    mapping = {
        -1: "Negative",
         0: "Neutral",
         1: "Positive"
    }
    return mapping.get(n, "Unknown")

def preprocess_data(df, vectorizer):
    x = vectorizer.fit_transform(df["text"])
    df['label'] = df['label'].astype(int)
    y = df["label"]
    return x, y

def preprocess_test_data(df, vectorizer):
    x = vectorizer.transform(df["text"])
    df['label'] = df['label'].astype(int)
    y = df["label"]
    return x, y