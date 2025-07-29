#!/usr/bin/env python3
"""
Watson AI integration for sentiment analysis
"""

from typing import Optional, Tuple, Dict
import warnings

# Watson SDK imports with fallback
try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False


class WatsonClient:
    """Watson Natural Language Understanding client"""

    def __init__(self, api_key: str, url: str):
        """Initialize Watson client

        Args:
            api_key: Watson API key
            url: Watson service URL

        Raises:
            ImportError: If Watson SDK is not available
            ValueError: If credentials are invalid
        """
        if not WATSON_AVAILABLE:
            raise ImportError(
                "IBM Watson SDK not available. Install with: pip install ibm-watson"
            )

        self.api_key = api_key
        self.url = url
        self.service = None
        self._setup_service()

    def _setup_service(self):
        """Setup Watson service connection"""
        try:
            authenticator = IAMAuthenticator(self.api_key)
            self.service = NaturalLanguageUnderstandingV1(
                version='2021-08-01',
                authenticator=authenticator
            )
            self.service.set_service_url(self.url)
            print("✅ Watson AI connected successfully!")
        except Exception as e:
            print(f"⚠️  Watson setup failed: {e}")
            self.service = None
            raise

    def get_sentiment(self, text: str) -> Optional[Tuple[int, float]]:
        """Get sentiment from Watson AI

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_label, confidence) or None if failed
            sentiment_label: 0=Negative, 1=Neutral, 2=Positive
            confidence: Float between 0 and 1
        """
        if not self.service:
            return None

        try:
            # Limit text length to avoid API limits
            if len(text) > 50000:
                text = text[:50000]

            response = self.service.analyze(
                text=text,
                features=Features(sentiment=SentimentOptions())
            ).get_result()

            sentiment = response['sentiment']['document']
            label = sentiment['label']  # positive, negative, neutral
            score = sentiment['score']  # -1 to 1

            # Convert to our format
            if label == 'positive':
                return 2, abs(score)
            elif label == 'negative':
                return 0, abs(score)
            else:
                return 1, abs(score)

        except Exception as e:
            print(f"Watson API error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Watson service is available"""
        return self.service is not None

    def test_connection(self) -> bool:
        """Test Watson API connection

        Returns:
            True if connection successful, False otherwise
        """
        if not self.service:
            return False

        try:
            # Test with simple text
            result = self.get_sentiment("This is a test.")
            return result is not None
        except Exception as e:
            print(f"Watson connection test failed: {e}")
            return False


class WatsonLabelGenerator:
    """Generates training labels using Watson AI"""

    def __init__(self, watson_client: Optional[WatsonClient] = None):
        """Initialize label generator

        Args:
            watson_client: Watson client instance (optional)
        """
        self.watson_client = watson_client

    def create_labels(self, texts: list, show_progress: bool = True) -> Tuple[list, list]:
        """Create training labels for texts

        Args:
            texts: List of texts to label
            show_progress: Whether to show progress updates

        Returns:
            Tuple of (labels, confidences)
        """
        labels = []
        confidences = []

        if show_progress:
            print(f"🎯 Creating training labels for {len(texts)} texts...")

        for i, text in enumerate(texts):
            # Progress indicator
            if show_progress and (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(texts)}...")

            # Use Watson if available
            if self.watson_client and self.watson_client.is_available():
                watson_result = self.watson_client.get_sentiment(text)
                if watson_result:
                    label, confidence = watson_result
                    labels.append(label)
                    confidences.append(confidence)
                    continue

            # Fallback to rule-based (will be handled by caller)
            labels.append(None)
            confidences.append(None)

        return labels, confidences

    def get_watson_stats(self, labels: list, confidences: list) -> Dict:
        """Get statistics about Watson labeling

        Args:
            labels: List of labels
            confidences: List of confidences

        Returns:
            Dictionary with Watson usage statistics
        """
        watson_count = sum(1 for label in labels if label is not None)
        total_count = len(labels)

        if watson_count > 0:
            avg_confidence = sum(c for c in confidences if c is not None) / watson_count
        else:
            avg_confidence = 0.0

        return {
            'total_texts': total_count,
            'watson_labeled': watson_count,
            'rule_based_labeled': total_count - watson_count,
            'watson_percentage': (watson_count / total_count) * 100 if total_count > 0 else 0,
            'average_confidence': avg_confidence
        }


def check_watson_availability() -> bool:
    """Check if Watson SDK is available

    Returns:
        True if Watson SDK is available, False otherwise
    """
    if WATSON_AVAILABLE:
        print("✅ IBM Watson SDK available")
        return True
    else:
        print("⚠️  IBM Watson SDK not installed. Using local processing only.")
        print("   Install with: pip install ibm-watson")
        return False


def create_watson_client(config: Dict) -> Optional[WatsonClient]:
    """Create Watson client from configuration

    Args:
        config: Dictionary with 'api_key' and 'url' keys

    Returns:
        WatsonClient instance or None if creation failed
    """
    if not WATSON_AVAILABLE:
        return None

    try:
        return WatsonClient(
            api_key=config['api_key'],
            url=config.get('url', 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com')
        )
    except Exception as e:
        print(f"Failed to create Watson client: {e}")
        return None