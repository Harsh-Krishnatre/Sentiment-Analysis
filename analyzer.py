#!/usr/bin/env python3
"""
Main sentiment analyzer that orchestrates all components
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')

from config import Config, WatsonConfig
from text_processor import TextProcessor, DataProcessor
from watson_client import WatsonClient, check_watson_availability, create_watson_client
from sentiment_model import SentimentModel
from utils import ResultsManager, SummaryGenerator


class SentimentAnalyzer:
    """Main sentiment analyzer class that orchestrates all components"""

    def __init__(self, use_watson: bool = False, config_file: str = 'config.json'):
        """Initialize the sentiment analyzer

        Args:
            use_watson: Whether to use Watson AI for training
            config_file: Path to configuration file
        """
        print("🚀 Initializing Sentiment Analyzer...")

        # Load configuration
        self.config = Config(config_file)

        # Initialize components
        self.text_processor = TextProcessor(
            **self.config.get_text_processing_config()
        )

        self.model = SentimentModel(
            vectorizer_params=self.config.get_vectorizer_params(),
            model_params=self.config.get_model_params()
        )

        self.data_processor = DataProcessor()
        self.results_manager = ResultsManager()
        self.summary_generator = SummaryGenerator()

        # Watson setup
        self.use_watson = use_watson and check_watson_availability()
        self.watson_client = None

        if self.use_watson:
            self._setup_watson()

        print("✅ Analyzer initialized successfully!")

    def _setup_watson(self):
        """Setup Watson AI client"""
        watson_config = self.config.get_watson_config()

        if watson_config:
            self.watson_client = create_watson_client(watson_config)
        else:
            print("⚠️  Watson config not found. Interactive setup required.")
            watson_config_manager = WatsonConfig()
            watson_config = watson_config_manager.setup_interactive()

            if watson_config:
                self.config.set_watson_config(
                    watson_config['api_key'],
                    watson_config.get('url')
                )
                self.config.save_config()
                self.watson_client = create_watson_client(watson_config)

    def train_model(self, texts: List[str], labels: Optional[List[int]] = None) -> Dict:
        """Train the sentiment model

        Args:
            texts: List of training texts
            labels: Optional pre-existing labels

        Returns:
            Training results dictionary
        """
        return self.model.train(
            texts=texts,
            labels=labels,
            watson_client=self.watson_client
        )

    def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single text

        Args:
            text: Text to analyze

        Returns:
            Prediction results dictionary
        """
        return self.model.predict(text)

    def analyze_csv(self, csv_file: str, text_column: Optional[str] = None,
                    output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Analyze sentiment in CSV file

        Args:
            csv_file: Path to CSV file
            text_column: Name of text column (auto-detected if None)
            output_file: Output file path (auto-generated if None)

        Returns:
            DataFrame with results or None if failed
        """
        print(f"📂 Loading CSV file: {csv_file}")

        # Load data
        try:
            df = self.data_processor.load_csv(csv_file)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None

        # Auto-detect text column if not specified
        if text_column is None:
            text_column = self.data_processor.detect_text_column(df)
            print(f"🔍 Using text column: '{text_column}'")

        # Validate text column
        if not self.data_processor.validate_text_column(df, text_column):
            return None

        # Train model if not trained
        if not self.model.is_trained:
            print("🎯 Training model on your data...")
            valid_texts = df[text_column].dropna().tolist()
            self.train_model(valid_texts)

        # Analyze sentiments
        texts = df[text_column].tolist()
        results = self.model.predict_batch(texts)

        # Add results to dataframe
        df['sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [round(r['confidence'], 3) for r in results]

        # Generate and print summary
        summary = self.summary_generator.generate_summary(df)
        self.summary_generator.print_summary(summary)

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sentiment_results_{timestamp}.csv"

        self.results_manager.save_csv_results(df, output_file)

        return df

    def analyze_texts(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict:
        """Analyze sentiment for a list of texts

        Args:
            texts: List of texts to analyze
            labels: Optional labels for the texts

        Returns:
            Dictionary with analysis results
        """
        # Train model if not trained
        if not self.model.is_trained:
            print("🎯 Training model on provided texts...")
            self.train_model(texts)

        # Analyze sentiments
        results = self.model.predict_batch(texts, show_progress=len(texts) > 10)

        # Create results dictionary
        analysis_results = {
            'predictions': results,
            'summary': self._create_text_summary(results, labels)
        }

        return analysis_results

    def _create_text_summary(self, results: List[Dict], labels: Optional[List[str]] = None) -> Dict:
        """Create summary for text analysis results

        Args:
            results: List of prediction results
            labels: Optional labels for texts

        Returns:
            Summary dictionary
        """
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results if r['sentiment'] != 'Unknown']

        # Count sentiments
        sentiment_counts = {
            'Positive': sentiments.count('Positive'),
            'Neutral': sentiments.count('Neutral'),
            'Negative': sentiments.count('Negative'),
            'Unknown': sentiments.count('Unknown')
        }

        total = len(results)
        sentiment_percentages = {
            k: (v / total) * 100 for k, v in sentiment_counts.items()
        }

        summary = {
            'total_texts': total,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': np.mean(confidences) if confidences else 0.0
        }

        if labels:
            summary['labels'] = labels

        return summary

    def get_model_info(self) -> Dict:
        """Get information about the current model

        Returns:
            Model information dictionary
        """
        info = self.model.get_model_info()
        info['watson_enabled'] = self.watson_client is not None
        info['watson_available'] = check_watson_availability()
        return info

    def save_model(self, filepath: str):
        """Save the trained model

        Args:
            filepath: Path to save the model
        """
        self.model.save_model(filepath)

    def load_model(self, filepath: str):
        """Load a trained model

        Args:
            filepath: Path to load the model from
        """
        self.model.load_model(filepath)

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get most important features for each sentiment class

        Args:
            top_n: Number of top features to return

        Returns:
            Feature importance dictionary
        """
        return self.model.get_feature_importance(top_n)

    def interactive_analysis(self):
        """Interactive mode for analyzing individual texts"""
        print("\n🎯 Interactive Sentiment Analysis")
        print("Type 'quit' to exit")
        print("-" * 40)

        # Train model if not trained
        if not self.model.is_trained:
            print("⚠️  Model not trained. Please analyze a CSV file first or provide training data.")
            return

        while True:
            text = input("\nEnter text to analyze: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not text:
                continue

            try:
                result = self.predict_sentiment(text)

                print(f"\nText: {text}")
                print(f"Sentiment: {result['sentiment']} "
                      f"(Confidence: {result['confidence']:.3f})")

                # Show probabilities
                probs = result['probabilities']
                print("Probabilities:")
                for sentiment, prob in probs.items():
                    print(f"  {sentiment}: {prob:.3f}")

            except Exception as e:
                print(f"❌ Error: {e}")