#!/usr/bin/env python3
"""
Command Line Interface for Basic Sentiment Analysis
"""

import argparse
import sys
import os
from typing import Optional

from analyzer import SentimentAnalyzer
from config import WatsonConfig
from utils import DataValidator, PerformanceMonitor, ResultsManager


def setup_watson_command():
    """Handle Watson setup command"""
    watson_config = WatsonConfig()
    config = watson_config.setup_interactive()

    if config:
        print("✅ Watson configuration completed!")
    else:
        print("⚠️  Watson configuration skipped.")


def analyze_csv_command(args):
    """Handle CSV analysis command"""
    # Validate input file
    if not DataValidator.validate_csv_file(args.csv_file):
        return False

    # Initialize performance monitor
    monitor = PerformanceMonitor()
    monitor.start_timer("total_analysis")

    # Initialize analyzer
    analyzer = SentimentAnalyzer(
        use_watson=args.watson,
        config_file=getattr(args, 'config', 'config.json')
    )

    # Analyze CSV
    try:
        results = analyzer.analyze_csv(
            csv_file=args.csv_file,
            text_column=args.column,
            output_file=args.output
        )

        if results is not None:
            # Show sample results
            text_col = analyzer.data_processor.detect_text_column(results)
            print(f"\nSample Results:")
            print("-" * 60)
            for i in range(min(3, len(results))):
                row = results.iloc[i]
                text = str(row[text_col])
                if len(text) > 50:
                    text = text[:50] + "..."
                print(f"Text: {text}")
                print(f"Sentiment: {row['sentiment']} (Confidence: {row['confidence']:.3f})")
                print()

            # Generate detailed report if requested
            if args.report:
                report_file = args.report
                if not report_file.endswith('.txt'):
                    report_file += '.txt'
                analyzer.summary_generator.save_report(results, text_col, report_file)

            # Save model if requested
            if args.save_model:
                analyzer.save_model(args.save_model)

            monitor.end_timer()

            if args.verbose:
                monitor.print_metrics()

            print("\n✅ Analysis completed successfully!")
            return True

        else:
            print("❌ Analysis failed!")
            return False

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def analyze_text_command(args):
    """Handle single text analysis command"""
    # Initialize analyzer
    analyzer = SentimentAnalyzer(
        use_watson=args.watson,
        config_file=getattr(args, 'config', 'config.json')
    )

    # Load model if specified
    if args.load_model:
        try:
            analyzer.load_model(args.load_model)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    if not analyzer.model.is_trained:
        print("❌ Model not trained. Please train a model first or load a pre-trained model.")
        return False

    try:
        result = analyzer.predict_sentiment(args.text)

        print(f"\nText: {args.text}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")

        if args.verbose:
            print("\nProbabilities:")
            for sentiment, prob in result['probabilities'].items():
                print(f"  {sentiment}: {prob:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing text: {e}")
        return False


def interactive_command(args):
    """Handle interactive mode command"""
    analyzer = SentimentAnalyzer(
        use_watson=args.watson,
        config_file=getattr(args, 'config', 'config.json')
    )

    # Load model if specified
    if args.load_model:
        try:
            analyzer.load_model(args.load_model)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    analyzer.interactive_analysis()
    return True


def model_info_command(args):
    """Handle model info command"""
    analyzer = SentimentAnalyzer(config_file=getattr(args, 'config', 'config.json'))

    # Load model if specified
    if args.load_model:
        try:
            analyzer.load_model(args.load_model)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    info = analyzer.get_model_info()

    print("\n🔍 Model Information:")
    print("-" * 30)
    print(f"Trained: {'Yes' if info['is_trained'] else 'No'}")
    print(f"Watson Available: {'Yes' if info['watson_available'] else 'No'}")
    print(f"Watson Enabled: {'Yes' if info['watson_enabled'] else 'No'}")

    if info['is_trained']:
        print(f"Vocabulary Size: {info['vocabulary_size']}")
        print(f"Feature Count: {info['feature_count']}")

    print(f"Sentiment Classes: {', '.join(info['sentiment_classes'])}")

    # Show feature importance if requested and model is trained
    if args.features and info['is_trained']:
        print("\n🔑 Top Features by Sentiment:")
        print("-" * 35)

        try:
            features = analyzer.get_feature_importance(args.features)
            for sentiment, feature_list in features.items():
                print(f"\n{sentiment}:")
                for i, (feature, score) in enumerate(feature_list[:10]):
                    print(f"  {i + 1:2d}. {feature} ({score:.3f})")
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")

    return True


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Basic Sentiment Analysis with Watson AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CSV file
  python main.py csv data.csv --watson --output results.csv

  # Analyze single text
  python main.py text "This is great!" --load-model model.pkl

  # Interactive mode
  python main.py interactive --load-model model.pkl

  # Setup Watson configuration
  python main.py setup-watson

  # Show model information
  python main.py info --load-model model.pkl --features 15
        """
    )

    # Global arguments
    parser.add_argument('--config', help='Configuration file path', default='config.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # CSV analysis command
    csv_parser = subparsers.add_parser('csv', help='Analyze CSV file')
    csv_parser.add_argument('csv_file', help='CSV file to analyze')
    csv_parser.add_argument('--column', '-c', help='Text column name')
    csv_parser.add_argument('--output', '-o', help='Output file name')
    csv_parser.add_argument('--watson', action='store_true', help='Use Watson AI for training')
    csv_parser.add_argument('--save-model', help='Save trained model to file')
    csv_parser.add_argument('--report', help='Generate detailed report file')

    # Single text analysis command
    text_parser = subparsers.add_parser('text', help='Analyze single text')
    text_parser.add_argument('text', help='Text to analyze')
    text_parser.add_argument('--watson', action='store_true', help='Use Watson AI')
    text_parser.add_argument('--load-model', help='Load pre-trained model')

    # Interactive mode command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive analysis mode')
    interactive_parser.add_argument('--watson', action='store_true', help='Use Watson AI')
    interactive_parser.add_argument('--load-model', help='Load pre-trained model')

    # Watson setup command
    subparsers.add_parser('setup-watson', help='Setup Watson AI configuration')

    # Model info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--load-model', help='Load model to inspect')
    info_parser.add_argument('--features', type=int, default=10,
                             help='Number of top features to show (default: 10)')

    return parser


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute commands
    success = False

    try:
        if args.command == 'csv':
            success = analyze_csv_command(args)
        elif args.command == 'text':
            success = analyze_text_command(args)
        elif args.command == 'interactive':
            success = interactive_command(args)
        elif args.command == 'setup-watson':
            setup_watson_command()
            success = True
        elif args.command == 'info':
            success = model_info_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())