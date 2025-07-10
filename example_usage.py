"""
Example Usage of Modular STS System
===================================

This script demonstrates how to use the modular STS system programmatically.
"""

from data.dataset_loader import STSDatasetLoader
from features.similarity_features import SimilarityFeatures
from features.text_preprocessor import TextPreprocessor
from models.neural_network import NeuralSTSModel
from evaluation.metrics import STSEvaluator
import numpy as np


def example_text_preprocessing():
    """Example of using text preprocessing."""
    print("="*50)
    print("EXAMPLE: Text Preprocessing")
    print("="*50)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example sentences
    sentences = [
        "A man is playing guitar and singing loudly!",
        "The cat is sleeping peacefully on the couch.",
        "I love eating pizza with my friends.",
        "The weather is nice today, isn't it?",
        "I hate vegetables but I eat them anyway."
    ]
    
    print("Original sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")
    
    # Preprocess sentences
    preprocessed = preprocessor.preprocess_sentences(sentences)
    
    print("\nPreprocessed sentences:")
    for i, sentence in enumerate(preprocessed, 1):
        print(f"  {i}. {sentence}")
    
    # Show tokenization example
    print("\nTokenization example:")
    tokens = preprocessor.tokenize_sentence("Hello, world! How are you?")
    print(f"  Tokens: {tokens}")
    
    # Show lemmatization example
    print("\nLemmatization example:")
    lemmatized = preprocessor.lemmatize_word("running", "v")
    print(f"  'running' -> '{lemmatized}'")
    
    # Show stop word removal
    print("\nStop word removal example:")
    words = ["the", "cat", "is", "sleeping", "on", "the", "mat"]
    filtered = preprocessor.remove_stop_words(words)
    print(f"  Original: {words}")
    print(f"  Filtered: {filtered}")


def example_traditional_features():
    """Example of using traditional similarity features."""
    print("\n" + "="*50)
    print("EXAMPLE: Traditional Similarity Features")
    print("="*50)
    
    # Initialize feature extractor
    feature_extractor = SimilarityFeatures()
    
    # Example sentence pairs
    sentence_pairs = [
        ("A man is playing guitar.", "A person is playing a musical instrument."),
        ("The cat is sleeping.", "A dog is running."),
        ("I love pizza.", "I enjoy eating pizza."),
        ("The weather is nice today.", "It's a beautiful day."),
        ("I hate vegetables.", "I dislike eating vegetables.")
    ]
    
    print("Extracting features for sentence pairs...")
    for i, (s1, s2) in enumerate(sentence_pairs, 1):
        features = feature_extractor.extract_all_features(s1, s2)
        print(f"\nPair {i}:")
        print(f"  S1: {s1}")
        print(f"  S2: {s2}")
        print(f"  Features: {len(features)} dimensions")
        print(f"  Sample features: {features[:5]}")


def example_neural_network():
    """Example of using neural network model."""
    print("\n" + "="*50)
    print("EXAMPLE: Neural Network Model")
    print("="*50)
    
    # Initialize neural model
    neural_model = NeuralSTSModel(input_size=14, hidden_sizes=[64, 32])
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Generate random features and scores
    features = np.random.randn(n_samples, 14)
    scores = np.random.uniform(0, 5, n_samples)
    
    # Split into train/dev
    train_size = int(0.8 * n_samples)
    train_features = features[:train_size]
    train_scores = scores[:train_size]
    dev_features = features[train_size:]
    dev_scores = scores[train_size:]
    
    print(f"Training neural network on {len(train_features)} samples...")
    print(f"Validation on {len(dev_features)} samples...")
    
    # Train model
    neural_model.train(train_features, train_scores, dev_features, dev_scores, epochs=5)
    
    # Make predictions
    predictions = neural_model.predict(dev_features)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")


def example_evaluation():
    """Example of using evaluation metrics."""
    print("\n" + "="*50)
    print("EXAMPLE: Evaluation Metrics")
    print("="*50)
    
    # Initialize evaluator
    evaluator = STSEvaluator()
    
    # Create dummy predictions
    np.random.seed(42)
    true_scores = np.random.uniform(0, 5, 100)
    predicted_scores = true_scores + np.random.normal(0, 0.5, 100)  # Add some noise
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(true_scores, predicted_scores)
    
    # Print results
    evaluator.print_metrics(metrics, "Example Dataset")
    
    # Analyze predictions
    sentences1 = [f"Sentence {i} A" for i in range(100)]
    sentences2 = [f"Sentence {i} B" for i in range(100)]
    
    analysis = evaluator.analyze_predictions(
        true_scores, predicted_scores, sentences1, sentences2, num_examples=3
    )
    
    evaluator.print_prediction_analysis(analysis, num_examples=2)


def example_dataset_loading():
    """Example of loading and analyzing datasets."""
    print("\n" + "="*50)
    print("EXAMPLE: Dataset Loading")
    print("="*50)
    
    # Initialize dataset loader
    loader = STSDatasetLoader()
    
    try:
        # Try to load datasets
        datasets = loader.load_all_datasets()
        
        if datasets:
            print("Successfully loaded datasets!")
            
            # Print dataset information
            loader.print_dataset_info(datasets)
            
            # Get sample pairs
            loader.print_sample_pairs(datasets, num_samples=2)
            
        else:
            print("No datasets found. Please run download_sts_dataset.py first.")
            
    except FileNotFoundError:
        print("Dataset files not found. Please run download_sts_dataset.py first.")


def example_complete_pipeline():
    """Example of a complete STS pipeline."""
    print("\n" + "="*50)
    print("EXAMPLE: Complete Pipeline")
    print("="*50)
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic sentence pairs and scores
    sentences1 = [f"This is sentence {i} with some content." for i in range(n_samples)]
    sentences2 = [f"This is sentence {i} with similar content." for i in range(n_samples)]
    
    # Generate realistic scores (some correlation between similar sentences)
    base_scores = np.random.uniform(0, 5, n_samples)
    # Add some correlation for similar sentences
    scores = np.clip(base_scores + np.random.normal(0, 1, n_samples), 0, 5)
    
    # Split data
    train_size = int(0.7 * n_samples)
    dev_size = int(0.15 * n_samples)
    
    train_s1 = sentences1[:train_size]
    train_s2 = sentences2[:train_size]
    train_scores = scores[:train_size]
    
    dev_s1 = sentences1[train_size:train_size + dev_size]
    dev_s2 = sentences2[train_size:train_size + dev_size]
    dev_scores = scores[train_size:train_size + dev_size]
    
    test_s1 = sentences1[train_size + dev_size:]
    test_s2 = sentences2[train_size + dev_size:]
    test_scores = scores[train_size + dev_size:]
    
    print(f"Training: {len(train_s1)} pairs")
    print(f"Development: {len(dev_s1)} pairs")
    print(f"Test: {len(test_s1)} pairs")
    
    # Initialize components
    preprocessor = TextPreprocessor()
    feature_extractor = SimilarityFeatures()
    evaluator = STSEvaluator()
    
    # Preprocess sentences
    print("\nPreprocessing sentences...")
    train_s1, train_s2 = preprocessor.preprocess_sentences(train_s1), preprocessor.preprocess_sentences(train_s2)
    dev_s1, dev_s2 = preprocessor.preprocess_sentences(dev_s1), preprocessor.preprocess_sentences(dev_s2)
    test_s1, test_s2 = preprocessor.preprocess_sentences(test_s1), preprocessor.preprocess_sentences(test_s2)
    
    # Extract features
    print("\nExtracting features...")
    train_features = []
    for s1, s2 in zip(train_s1, train_s2):
        features = feature_extractor.extract_all_features(s1, s2)
        train_features.append(features)
    
    dev_features = []
    for s1, s2 in zip(dev_s1, dev_s2):
        features = feature_extractor.extract_all_features(s1, s2)
        dev_features.append(features)
    
    test_features = []
    for s1, s2 in zip(test_s1, test_s2):
        features = feature_extractor.extract_all_features(s1, s2)
        test_features.append(features)
    
    train_features = np.array(train_features)
    dev_features = np.array(dev_features)
    test_features = np.array(test_features)
    
    print(f"Feature dimensions: {train_features.shape[1]}")
    
    # Train neural model
    print("\nTraining neural network...")
    neural_model = NeuralSTSModel(input_size=train_features.shape[1], hidden_sizes=[64, 32])
    neural_model.train(train_features, train_scores, dev_features, dev_scores, epochs=10)
    
    # Make predictions
    print("\nMaking predictions...")
    test_predictions = neural_model.predict(test_features)
    
    # Evaluate
    print("\nEvaluating results...")
    metrics = evaluator.calculate_all_metrics(test_scores, test_predictions)
    evaluator.print_metrics(metrics, "Test Set")
    
    print("\nPipeline completed successfully!")


def main():
    """Run all examples."""
    print("MODULAR STS SYSTEM - EXAMPLE USAGE")
    print("="*60)
    
    # Run examples
    example_text_preprocessing()
    example_traditional_features()
    example_neural_network()
    example_evaluation()
    example_dataset_loading()
    example_complete_pipeline()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)
    print("\nTo run the full system:")
    print("  python sts_similarity_modular.py")
    print("\nTo run with BERT:")
    print("  python sts_similarity_modular.py --use-bert")
    print("\nTo run interactively:")
    print("  python sts_similarity_modular.py --interactive")


if __name__ == "__main__":
    main() 