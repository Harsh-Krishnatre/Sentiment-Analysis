import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
import joblib
from utils.preprocess import stringify
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["review"]
        if user_input.strip():
            # Convert input to DataFrame for compatibility
            df = pd.DataFrame({"text": [user_input]})
            x = vectorizer.transform(df["text"])
            y_pred = model.predict(x)[0]
            prediction = stringify(y_pred)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
