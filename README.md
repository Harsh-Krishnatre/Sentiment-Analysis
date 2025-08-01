# 🧠 Sentiment Analysis with Naive Bayes (Text Classification)

This project is a simple yet modular implementation of a **Sentiment Analysis** system using the **Naive Bayes classifier**. It takes labeled text data and predicts whether the sentiment is **positive (1)**, or **negative (0)**.

---

## 📂 Project Structure

```
Sentiment-Analysis/
│
├── data/                  # Sample test and training CSV files
│   ├── sample_test.csv
│   └── sample_train.csv
|   |__ sample_train_large.csv
│
├── model/                 # ML model training and testing logic
│   ├── train.py
│   ├── test.py
|   |__ __init__.py
│
|__ utils/
|   |__ preprocess.py
|   |__ __init__.py
│
│__ frontend/
│   │__ templates/
│   │    │__ index.html
│   │__ app.py
│
│              
│── model.pkl          # Trained Naive Bayes model
│__ vectorizer.pkl
│
├── main.py                # Main user interface script
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

---

## 🚀 Features

- Text preprocessing using `TFidVectorizer`
- Modular and readable code
- Training & evaluation separated cleanly
- User-friendly terminal interface
- Prediction result preview + classification report

---

## 🔧 Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Input Format

CSV file should have at least these columns:

```csv
Serial No.,text,label
1,"I loved the movie",positive
2,"It was boring",negative
3,"The product is fine",positive
```

Where:
- `1` = Positive
- `0` = Negative

---

## 🏃‍♂️ Running the Project

### To start the CLI:

```bash
python main.py
```

Follow the prompt to:

- Train model from a CSV
- Test the model on another CSV
- See predictions and evaluation report

---

## 📊 Example Output

```
=== Training Report ===
              precision    recall  f1-score   support

          -1       0.86      0.87      0.87       50
           0       0.71      0.69      0.70       50
           1       0.92      0.94      0.93       50

    accuracy                           0.83      150
```

---

## ✅ Dependencies

- `pandas`
- `scikit-learn`
- `joblib`

Install them all using:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Make sure to **use the same vectorizer** used during training for testing.
- You can expand this to include deep learning models, more preprocessing, or custom visualizations.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).