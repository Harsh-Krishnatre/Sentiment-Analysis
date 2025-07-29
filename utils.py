#!/usr/bin/env python3
"""
Utility functions for sentiment analysis
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class ResultsManager:
    """Manages saving and loading of analysis results"""

    @staticmethod
    def save_csv_results(df: pd.DataFrame, output_file: str):
        """Save DataFrame results to CSV

        Args:
            df: DataFrame with results
            output_file: Output file path
        """
        try:
            df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    @staticmethod
    def save_json_results(results: Dict, output_file: str):
        """Save results dictionary to JSON

        Args:
            results: Results dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving JSON results: {e}")

    @staticmethod
    def load_json_results(input_file: str) -> Optional[Dict]:
        """Load results from JSON file

        Args:
            input_file: Input file path

        Returns:
            Results dictionary or None if failed
        """
        try:
            with open(input_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading JSON results: {e}")
            return None

    @staticmethod
    def create_output_filename(base_name: str, extension: str = '.csv') -> str:
        """Create timestamped output filename

        Args:
            base_name: Base name for the file
            extension: File extension

        Returns:
            Timestamped filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}{extension}"


class SummaryGenerator:
    """Generates analysis summaries and reports"""

    def generate_summary(self, df: pd.DataFrame,
                         sentiment_col: str = 'sentiment',
                         confidence_col: str = 'confidence') -> Dict:
        """Generate summary statistics for sentiment analysis results

        Args:
            df: DataFrame with results
            sentiment_col: Name of sentiment column
            confidence_col: Name of confidence column

        Returns:
            Summary statistics dictionary
        """
        # Basic counts
        counts = df[sentiment_col].value_counts().to_dict()
        total = len(df)

        # Percentages
        percentages = {k: (v / total) * 100 for k, v in counts.items()}

        # Confidence statistics
        valid_confidence = df[df[sentiment_col] != 'Unknown'][confidence_col]
        confidence_stats = {
            'mean': valid_confidence.mean() if len(valid_confidence) > 0 else 0.0,
            'std': valid_confidence.std() if len(valid_confidence) > 0 else 0.0,
            'min': valid_confidence.min() if len(valid_confidence) > 0 else 0.0,
            'max': valid_confidence.max() if len(valid_confidence) > 0 else 0.0
        }

        # High confidence predictions
        high_confidence_threshold = 0.8
        high_conf_mask = valid_confidence >= high_confidence_threshold
        high_confidence_count = high_conf_mask.sum()
        high_confidence_percentage = (high_confidence_count / len(valid_confidence) * 100) if len(
            valid_confidence) > 0 else 0.0

        return {
            'total_texts': total,
            'sentiment_counts': counts,
            'sentiment_percentages': percentages,
            'confidence_stats': confidence_stats,
            'high_confidence_count': high_confidence_count,
            'high_confidence_percentage': high_confidence_percentage,
            'high_confidence_threshold': high_confidence_threshold
        }

    def print_summary(self, summary: Dict):
        """Print formatted summary statistics

        Args:
            summary: Summary dictionary from generate_summary()
        """
        print("\n" + "=" * 50)
        print("📊 SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)

        # Sentiment counts
        counts = summary['sentiment_counts']
        percentages = summary['sentiment_percentages']
        total = summary['total_texts']

        for sentiment in ['Positive', 'Neutral', 'Negative', 'Unknown']:
            count = counts.get(sentiment, 0)
            percentage = percentages.get(sentiment, 0.0)
            print(f"{sentiment:>10}: {count:>6} ({percentage:>5.1f}%)")

        print(f"{'Total':>10}: {total:>6} (100.0%)")

        # Confidence statistics
        conf_stats = summary['confidence_stats']
        if conf_stats['mean'] > 0:
            print(f"\nConfidence Statistics:")
            print(f"  Average: {conf_stats['mean']:.3f}")
            print(f"  Std Dev: {conf_stats['std']:.3f}")
            print(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")

            # High confidence predictions
            high_conf_count = summary['high_confidence_count']
            high_conf_pct = summary['high_confidence_percentage']
            threshold = summary['high_confidence_threshold']
            print(f"  High Confidence (≥{threshold}): {high_conf_count} ({high_conf_pct:.1f}%)")

        print("=" * 50)

    def generate_detailed_report(self, df: pd.DataFrame,
                                 text_col: str,
                                 sentiment_col: str = 'sentiment',
                                 confidence_col: str = 'confidence') -> str:
        """Generate detailed analysis report

        Args:
            df: DataFrame with results
            text_col: Name of text column
            sentiment_col: Name of sentiment column
            confidence_col: Name of confidence column

        Returns:
            Formatted report string
        """
        summary = self.generate_summary(df, sentiment_col, confidence_col)

        report_lines = [
            "DETAILED SENTIMENT ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Texts Analyzed: {summary['total_texts']}",
            "",
            "SENTIMENT DISTRIBUTION:",
            "-" * 25
        ]

        # Add sentiment distribution
        for sentiment, count in summary['sentiment_counts'].items():
            percentage = summary['sentiment_percentages'][sentiment]
            report_lines.append(f"{sentiment}: {count} ({percentage:.1f}%)")

        # Add confidence analysis
        if summary['confidence_stats']['mean'] > 0:
            report_lines.extend([
                "",
                "CONFIDENCE ANALYSIS:",
                "-" * 20,
                f"Average Confidence: {summary['confidence_stats']['mean']:.3f}",
                f"Standard Deviation: {summary['confidence_stats']['std']:.3f}",
                f"Confidence Range: {summary['confidence_stats']['min']:.3f} - {summary['confidence_stats']['max']:.3f}",
                f"High Confidence Predictions (≥{summary['high_confidence_threshold']}): "
                f"{summary['high_confidence_count']} ({summary['high_confidence_percentage']:.1f}%)"
            ])

        # Add sample results
        report_lines.extend([
            "",
            "SAMPLE RESULTS:",
            "-" * 15
        ])

        # Show samples from each sentiment category
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            sentiment_samples = df[df[sentiment_col] == sentiment].head(3)
            if len(sentiment_samples) > 0:
                report_lines.append(f"\n{sentiment} Examples:")
                for _, row in sentiment_samples.iterrows():
                    text = str(row[text_col])
                    if len(text) > 100:
                        text = text[:97] + "..."
                    confidence = row[confidence_col]
                    report_lines.append(f"  • {text} (conf: {confidence:.3f})")

        return "\n".join(report_lines)

    def save_report(self, df: pd.DataFrame, text_col: str, output_file: str):
        """Save detailed report to file

        Args:
            df: DataFrame with results
            text_col: Name of text column
            output_file: Output file path
        """
        report = self.generate_detailed_report(df, text_col)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")


class DataValidator:
    """Validates input data and parameters"""

    @staticmethod
    def validate_csv_file(file_path: str) -> bool:
        """Validate CSV file exists and is readable

        Args:
            file_path: Path to CSV file

        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        if not file_path.lower().endswith('.csv'):
            print(f"⚠️  File doesn't appear to be a CSV: {file_path}")

        try:
            # Try to read first few rows
            pd.read_csv(file_path, nrows=5)
            return True
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            return False

    @staticmethod
    def validate_text_list(texts: List[str]) -> bool:
        """Validate list of texts

        Args:
            texts: List of texts to validate

        Returns:
            True if valid, False otherwise
        """
        if not texts:
            print("❌ Empty text list provided")
            return False

        if not isinstance(texts, list):
            print("❌ Texts must be provided as a list")
            return False

        # Check for valid text content
        valid_texts = [t for t in texts if t and str(t).strip()]
        if len(valid_texts) == 0:
            print("❌ No valid text content found")
            return False

        if len(valid_texts) < len(texts):
            print(f"⚠️  {len(texts) - len(valid_texts)} empty/invalid texts will be skipped")

        return True

    @staticmethod
    def validate_config(config: dict) -> bool:
        """Validate configuration dictionary

        Args:
            config: Configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        required_sections = ['vectorizer', 'model']

        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required config section: {section}")
                return False

        # Validate vectorizer params
        vectorizer_params = config['vectorizer']
        if 'max_features' in vectorizer_params and vectorizer_params['max_features'] <= 0:
            print("❌ max_features must be positive")
            return False

        # Validate model params
        model_params = config['model']
        if 'alpha' in model_params and model_params['alpha'] <= 0:
            print("❌ alpha must be positive")
            return False

        return True


class PerformanceMonitor:
    """Monitors and reports performance metrics"""

    def __init__(self):
        self.start_time = None
        self.metrics = {}

    def start_timer(self, operation: str):
        """Start timing an operation

        Args:
            operation: Name of the operation
        """
        self.start_time = datetime.now()
        self.current_operation = operation

    def end_timer(self) -> float:
        """End timing and return duration

        Returns:
            Duration in seconds
        """
        if self.start_time is None:
            return 0.0

        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics[self.current_operation] = duration
        self.start_time = None

        return duration

    def get_metrics(self) -> Dict:
        """Get all recorded metrics

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def print_metrics(self):
        """Print all recorded metrics"""
        if not self.metrics:
            print("No performance metrics recorded")
            return

        print("\n⏱️  Performance Metrics:")
        print("-" * 25)
        for operation, duration in self.metrics.items():
            print(f"{operation}: {duration:.2f} seconds")