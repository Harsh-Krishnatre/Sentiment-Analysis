#!/usr/bin/env python3
"""
Core sentiment analysis model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from text_processor import TextProcessor
from watson_client import WatsonClient, WatsonLabelGenerator


class SentimentModel:
    """Core sentiment analysis model"""

    def __init__(self, vectorizer_params: Dict = None, model_params: Dict = None):
        """Initialize sentiment model

        Args:
            vectorizer_params: Parameters for TfidfVectorizer
            model_params: Parameters for MultinomialNB
        """
        # Default parameters
        self.vectorizer_params = vectorizer_params or {
            'max_features': 3000,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }

        self.model_params = model_params or {'alpha': 0.5}

        # Initialize components
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.model = MultinomialNB(**self.model_params)

        # State
        self.is_trained = False
        self.sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

        # Components
        self.text_processor = TextProcessor()

    def train(self, texts: List[str], labels: Optional[List[int]] = None,
              watson_client: Optional[WatsonClient] = None,
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the sentiment model

        Args:
            texts: List of training texts
            labels: Optional pre-existing labels
            watson_client: Optional Watson client for labeling
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with training results
        """
        print("🤖 Training sentiment model...")

        # Clean texts
        cleaned_texts = self.text_processor.clean_texts(texts)

        # Generate labels if not provided
        if labels is None:
            labels = self._generate_labels(cleaned_texts, watson_client)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Train model
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

        # Evaluate
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        unique_classes = np.unique(y_test)

        # Generate report
        report = classification_report(
            y_test, y_pred,
            labels=unique_classes,
            target_names=['Negative', 'Neutral', 'Positive'],
            output_dict=True
        )



        print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels = unique_classes,target_names=['Negative', 'Neutral', 'Positive']))

        self.is_trained = True

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

    def _generate_labels(self, texts: List[str], watson_client: Optional[WatsonClient]) -> List[int]:
        """Generate labels for training texts

        Args:
            texts: List of cleaned texts
            watson_client: Optional Watson client

        Returns:
            List of sentiment labels
        """
        if watson_client:
            label_generator = WatsonLabelGenerator(watson_client)
            watson_labels, watson_confidences = label_generator.create_labels(texts)

            # Fill in None values with rule-based sentiment
            final_labels = []
            for i, (watson_label, text) in enumerate(zip(watson_labels, texts)):
                if watson_label is not None:
                    final_labels.append(watson_label)
                else:
                    rule_label = self.text_processor.rule_based_sentiment(text)
                    final_labels.append(rule_label)

            # Print Watson usage stats
            stats = label_generator.get_watson_stats(watson_labels, watson_confidences)
            print(f"Watson labeled: {stats['watson_labeled']}/{stats['total_texts']} "
                  f"({stats['watson_percentage']:.1f}%)")

            return final_labels
        else:
            # Use rule-based labeling only
            print("Using rule-based labeling...")
            return [self.text_processor.rule_based_sentiment(text) for text in texts]

    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with prediction results

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        cleaned_text = self.text_processor.clean_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])

        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]

        return {
            'text': text,
            'sentiment': self.sentiment_map[prediction],
            'confidence': max(probabilities),
            'probabilities': {
                'Negative': probabilities[0],
                'Neutral': probabilities[1],
                'Positive': probabilities[2]
            }
        }

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """Predict sentiments for multiple texts

        Args:
            texts: List of texts to analyze
            show_progress: Whether to show progress updates

        Returns:
            List of prediction dictionaries

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        results = []

        if show_progress:
            print("🔮 Analyzing sentiments...")

        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 200 == 0:
                print(f"   Processed {i + 1}/{len(texts)}...")

            if pd.isna(text):
                results.append({
                    'text': text,
                    'sentiment': 'Unknown',
                    'confidence': 0.0,
                    'probabilities': {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                })
            else:
                result = self.predict(text)
                results.append(result)

        return results

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get most important features for each sentiment class

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with top features for each class

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call get_feature_importance() after training.")

        feature_names = self.vectorizer.get_feature_names_out()

        # Get feature log probabilities for each class
        feature_importance = {}

        for class_idx, class_name in self.sentiment_map.items():
            # Get log probabilities for this class
            log_probs = self.model.feature_log_prob_[class_idx]

            # Get top features
            top_indices = np.argsort(log_probs)[-top_n:][::-1]
            top_features = [(feature_names[i], log_probs[i]) for i in top_indices]

            feature_importance[class_name] = top_features

        return feature_importance

    def save_model(self, filepath: str):
        """Save model to file

        Args:
            filepath: Path to save the model
        """
        import joblib

        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'sentiment_map': self.sentiment_map,
            'vectorizer_params': self.vectorizer_params,
            'model_params': self.model_params
        }

        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file

        Args:
            filepath: Path to load the model from
        """
        import joblib

        try:
            model_data = joblib.load(filepath)

            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.sentiment_map = model_data['sentiment_map']
            self.vectorizer_params = model_data.get('vectorizer_params', {})
            self.model_params = model_data.get('model_params', {})

            self.is_trained = True
            print(f"✅ Model loaded from {filepath}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the current model

        Returns:
            Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'vectorizer_params': self.vectorizer_params,
            'model_params': self.model_params,
            'sentiment_classes': list(self.sentiment_map.values())
        }

        if self.is_trained:
            info.update({
                'vocabulary_size': len(self.vectorizer.vocabulary_),
                'feature_count': len(self.vectorizer.get_feature_names_out())
            })

        return info