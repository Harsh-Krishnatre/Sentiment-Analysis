from flask import Flask, render_template, request
import re
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review = ""
    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            cleaned_review = clean_text(review)
            x = vectorizer.transform([cleaned_review])
            y_pred = model.predict(x)[0]
            prediction = y_pred
    return render_template("index.html", prediction=prediction, review=review)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 for local dev
    app.run(host="0.0.0.0", port=port)