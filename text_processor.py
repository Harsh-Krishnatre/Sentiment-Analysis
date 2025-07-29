#!/usr/bin/env python3
"""
Text processing utilities for sentiment analysis
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Optional


class TextProcessor:
    """Handles text cleaning and preprocessing"""

    def __init__(self, positive_words: Optional[List[str]] = None,
                 negative_words: Optional[List[str]] = None):
        """Initialize text processor

        Args:
            positive_words: List of positive sentiment words
            negative_words: List of negative sentiment words
        """
        self._setup_nltk()

        self.positive_words = positive_words or self._default_positive_words()
        self.negative_words = negative_words or self._default_negative_words()

    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("📥 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    def _default_positive_words(self) -> List[str]:
        """Default positive words for rule-based sentiment"""
        return [
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'wonderful', 'perfect', 'love', 'best', 'recommend', 'happy',
            'satisfied', 'quality', 'fast', 'easy', 'helpful'
        ]

    def _default_negative_words(self) -> List[str]:
        """Default negative words for rule-based sentiment"""
        return [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'useless', 'broken', 'poor', 'slow',
            'expensive', 'difficult', 'problem', 'issue', 'waste'
        ]

    def clean_text(self, text) -> str:
        """Clean and preprocess text

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def clean_texts(self, texts: List[str]) -> List[str]:
        """Clean multiple texts

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def rule_based_sentiment(self, text: str) -> int:
        """Simple rule-based sentiment detection

        Args:
            text: Text to analyze

        Returns:
            Sentiment label (0=Negative, 1=Neutral, 2=Positive)
        """
        text_lower = text.lower()

        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)

        if positive_count > negative_count:
            return 2  # Positive
        elif negative_count > positive_count:
            return 0  # Negative
        else:
            return 1  # Neutral

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        try:
            return word_tokenize(text.lower())
        except Exception:
            return text.lower().split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens without stopwords
        """
        try:
            stop_words = set(stopwords.words('english'))
            return [token for token in tokens if token not in stop_words]
        except Exception:
            # Fallback if NLTK stopwords not available
            basic_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                'through', 'during', 'before', 'after', 'above', 'below',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'then', 'once'
            }
            return [token for token in tokens if token not in basic_stopwords]

    def extract_features(self, text: str) -> dict:
        """Extract various text features

        Args:
            text: Text to analyze

        Returns:
            Dictionary of text features
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)

        return {
            'length': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'positive_word_count': sum(1 for word in self.positive_words if word in cleaned_text.lower()),
            'negative_word_count': sum(1 for word in self.negative_words if word in cleaned_text.lower())
        }


class DataProcessor:
    """Handles data loading and column detection"""

    @staticmethod
    def detect_text_column(df: pd.DataFrame) -> str:
        """Auto-detect the text column in a DataFrame

        Args:
            df: Pandas DataFrame

        Returns:
            Name of the detected text column
        """
        # Common column names
        text_cols = ['review', 'text', 'comment', 'feedback', 'description', 'content']

        for col in df.columns:
            if col.lower() in text_cols:
                return col

        # Find column with longest average text
        best_col = None
        max_avg_len = 0

        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    best_col = col

        return best_col or df.columns[0]

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load CSV file with error handling

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            pd.errors.ParserError: If file can't be parsed
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} records from {file_path}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Error parsing file {file_path}: {e}")

    @staticmethod
    def validate_text_column(df: pd.DataFrame, column_name: str) -> bool:
        """Validate that a text column exists and has data

        Args:
            df: DataFrame to check
            column_name: Name of column to validate

        Returns:
            True if column is valid, False otherwise
        """
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found!")
            print(f"Available columns: {list(df.columns)}")
            return False

        # Check if column has any non-null text data
        non_null_count = df[column_name].notna().sum()
        if non_null_count == 0:
            print(f"❌ Column '{column_name}' has no valid text data!")
            return False

        print(f"✅ Column '{column_name}' validated ({non_null_count} non-null entries)")
        return True