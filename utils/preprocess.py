def preprocess_data(df, vectorizer):
    x = vectorizer.fit_transform(df["review"])
    df['sentiment'] = df['sentiment'].astype(str)
    y = df["sentiment"]
    return x, y

def preprocess_test_data(df, vectorizer):
    x = vectorizer.transform(df["review"])
    df['sentiment'] = df['sentiment'].astype(str)
    y = df["sentiment"]
    return x, y