#!/usr/bin/env python3
"""
Configuration management for sentiment analysis
"""

import json
import os
from typing import Dict, Optional


class Config:
    """Configuration manager for the sentiment analyzer"""

    # Default settings
    DEFAULT_VECTORIZER_PARAMS = {
        'max_features': 3000,
        'stop_words': 'english',
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.8
    }

    DEFAULT_MODEL_PARAMS = {
        'alpha': 0.5
    }

    DEFAULT_WATSON_URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com'

    def __init__(self, config_file: str = 'config.json'):
        """Initialize configuration

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Error loading config: {e}. Using defaults.")

        return self._create_default_config()

    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        return {
            'vectorizer': self.DEFAULT_VECTORIZER_PARAMS,
            'model': self.DEFAULT_MODEL_PARAMS,
            'watson': {
                'url': self.DEFAULT_WATSON_URL
            },
            'text_processing': {
                'positive_words': [
                    'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
                    'wonderful', 'perfect', 'love', 'best', 'recommend', 'happy',
                    'satisfied', 'quality', 'fast', 'easy', 'helpful'
                ],
                'negative_words': [
                    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                    'disappointing', 'useless', 'broken', 'poor', 'slow',
                    'expensive', 'difficult', 'problem', 'issue', 'waste'
                ]
            }
        }

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except IOError as e:
            print(f"❌ Error saving config: {e}")

    def get_vectorizer_params(self) -> Dict:
        """Get vectorizer parameters"""
        return self.config.get('vectorizer', self.DEFAULT_VECTORIZER_PARAMS)

    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return self.config.get('model', self.DEFAULT_MODEL_PARAMS)

    def get_watson_config(self) -> Optional[Dict]:
        """Get Watson configuration"""
        watson_config = self.config.get('watson', {})
        if 'api_key' in watson_config:
            return watson_config
        return None

    def set_watson_config(self, api_key: str, url: Optional[str] = None):
        """Set Watson configuration

        Args:
            api_key: Watson API key
            url: Watson service URL (optional)
        """
        self.config['watson'] = {
            'api_key': api_key,
            'url': url or self.DEFAULT_WATSON_URL
        }

    def get_text_processing_config(self) -> Dict:
        """Get text processing configuration"""
        return self.config.get('text_processing', {})


class WatsonConfig:
    """Watson-specific configuration manager"""

    def __init__(self, config_file: str = 'watson_config.json'):
        self.config_file = config_file

    def setup_interactive(self) -> Optional[Dict]:
        """Interactive Watson configuration setup"""
        print("\n🔧 Watson AI Configuration Setup")
        print("=" * 40)
        print("To use Watson AI, you need:")
        print("1. IBM Cloud account (free tier available)")
        print("2. Natural Language Understanding service")
        print("3. API key from your service credentials")
        print()

        api_key = input("Enter your Watson API key (or press Enter to skip): ").strip()

        if api_key:
            url = input("Enter service URL (or press Enter for default): ").strip()
            config = {
                'api_key': api_key,
                'url': url or Config.DEFAULT_WATSON_URL
            }

            # Save Watson-specific config
            self.save_config(config)
            return config

        return None

    def save_config(self, config: Dict):
        """Save Watson configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Watson config saved to {self.config_file}")
        except IOError as e:
            print(f"❌ Error saving Watson config: {e}")

    def load_config(self) -> Optional[Dict]:
        """Load Watson configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Error loading Watson config: {e}")
        return None