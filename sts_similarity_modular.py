"""
Modular Semantic Textual Similarity (STS) System
================================================

This system provides multiple approaches for computing semantic similarity between sentences:
1. Traditional ML with handcrafted features
2. Neural Network with PyTorch
3. BERT embeddings

Usage:
    python sts_similarity_modular.py
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional

# Import our modular components
from data.dataset_loader import STSDatasetLoader
from features.similarity_features import SimilarityFeatures
from features.text_preprocessor import TextPreprocessor
from models.neural_network import NeuralSTSModel
from models.bert_embeddings import BERTFeatureExtractor
from evaluation.metrics import STSEvaluator

# Traditional ML imports
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class ModularSTSSystem:
    """Modular STS system with multiple approaches."""
    
    def __init__(self, use_bert: bool = False, use_neural: bool = True):
        """
        Initialize the modular STS system.
        
        Args:
            use_bert: Whether to use BERT embeddings
            use_neural: Whether to use neural network
        """
        self.use_bert = use_bert
        self.use_neural = use_neural
        
        # Initialize components
        self.data_loader = STSDatasetLoader()
        self.feature_extractor = SimilarityFeatures()
        self.preprocessor = TextPreprocessor()
        self.evaluator = STSEvaluator()
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        
        if use_bert:
            print("Initializing BERT feature extractor...")
            self.bert_extractor = BERTFeatureExtractor()
        
        if use_neural:
            print("Initializing neural network model...")
            self.neural_model = NeuralSTSModel()
    
    def load_data(self) -> Dict[str, Tuple[List[str], List[str], List[float]]]:
        """Load all datasets."""
        print("Loading datasets...")
        datasets = self.data_loader.load_all_datasets()
        self.data_loader.print_dataset_info(datasets)
        self.data_loader.print_sample_pairs(datasets)
        return datasets
    
    def preprocess_sentences(self, sentences1: List[str], sentences2: List[str]) -> Tuple[List[str], List[str]]:
        """Preprocess sentences using the text preprocessor."""
        print("Preprocessing sentences...")
        preprocessed_s1 = self.preprocessor.preprocess_sentences(sentences1)
        preprocessed_s2 = self.preprocessor.preprocess_sentences(sentences2)
        return preprocessed_s1, preprocessed_s2
    
    def extract_traditional_features(self, sentences1: List[str], 
                                   sentences2: List[str]) -> np.ndarray:
        """Extract traditional similarity features."""
        print("Extracting traditional features...")
        features = []
        
        for i, (s1, s2) in enumerate(zip(sentences1, sentences2)):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(sentences1)} pairs")
            
            feature_vector = self.feature_extractor.extract_all_features(
                s1, s2, self.data_loader.word_counts
            )
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_bert_features(self, sentences1: List[str], 
                            sentences2: List[str]) -> np.ndarray:
        """Extract BERT features."""
        if not self.use_bert:
            raise ValueError("BERT not enabled")
        
        print("Extracting BERT features...")
        sentence_pairs = list(zip(sentences1, sentences2))
        features = self.bert_extractor.extract_batch_features(sentence_pairs)
        return features
    
    def train_traditional_models(self, train_features: np.ndarray, 
                               train_scores: List[float],
                               dev_features: np.ndarray, 
                               dev_scores: List[float]):
        """Train traditional ML models."""
        print("Training traditional ML models...")
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        dev_features_scaled = scaler.transform(dev_features)
        
        # Train different models
        models_to_train = {
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        for model_name, model in models_to_train.items():
            print(f"  Training {model_name}...")
            start_time = time.time()
            
            model.fit(train_features_scaled, train_scores)
            
            # Make predictions
            train_pred = model.predict(train_features_scaled)
            dev_pred = model.predict(dev_features_scaled)
            
            # Calculate metrics
            train_metrics = self.evaluator.calculate_all_metrics(train_scores, train_pred)
            dev_metrics = self.evaluator.calculate_all_metrics(dev_scores, dev_pred)
            
            print(f"    Training time: {time.time() - start_time:.2f}s")
            print(f"    Dev Pearson: {dev_metrics['pearson']:.4f}")
            
            # Store model and scaler
            self.models[model_name] = model
            self.scalers[model_name] = scaler
    
    def train_neural_model(self, train_features: np.ndarray, 
                          train_scores: List[float],
                          dev_features: np.ndarray, 
                          dev_scores: List[float]):
        """Train neural network model."""
        if not self.use_neural:
            return
        
        print("Training neural network model...")
        
        # Scale features for neural network
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        dev_features_scaled = scaler.transform(dev_features)
        
        # Train neural network
        self.neural_model.train(
            train_features_scaled, train_scores,
            dev_features_scaled, dev_scores,
            epochs=30, patience=5
        )
        
        # Store scaler
        self.scalers['NeuralNetwork'] = scaler
    
    def predict_with_traditional_models(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with traditional models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            features_scaled = scaler.transform(features)
            predictions[model_name] = model.predict(features_scaled)
        
        return predictions
    
    def predict_with_neural_model(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with neural network."""
        if not self.use_neural:
            return None
        
        scaler = self.scalers['NeuralNetwork']
        features_scaled = scaler.transform(features)
        return self.neural_model.predict(features_scaled)
    
    def evaluate_models(self, datasets: Dict[str, Tuple[List[str], List[str], List[float]]]):
        """Evaluate all models on all datasets."""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        all_results = {}
        
        for split_name, (sentences1, sentences2, scores) in datasets.items():
            if not scores:  # Skip test set if no scores
                continue
            
            print(f"\nEvaluating on {split_name} set...")
            
            # Preprocess sentences
            sentences1, sentences2 = self.preprocess_sentences(sentences1, sentences2)
            
            # Extract features
            if self.use_bert:
                features = self.extract_bert_features(sentences1, sentences2)
            else:
                features = self.extract_traditional_features(sentences1, sentences2)
            
            # Get predictions from all models
            predictions = {}
            
            # Traditional models
            if self.models:
                trad_predictions = self.predict_with_traditional_models(features)
                predictions.update(trad_predictions)
            
            # Neural network
            if self.use_neural:
                neural_pred = self.predict_with_neural_model(features)
                if neural_pred is not None:
                    predictions['NeuralNetwork'] = neural_pred
            
            # Evaluate each model
            split_results = {}
            for model_name, pred_scores in predictions.items():
                metrics = self.evaluator.calculate_all_metrics(scores, pred_scores)
                split_results[model_name] = metrics
                self.evaluator.print_metrics(metrics, f"{split_name} - {model_name}")
            
            all_results[split_name] = split_results
        
        # Compare models
        if len(all_results) > 0:
            # Use dev set for comparison if available, otherwise use first available
            comparison_set = 'dev' if 'dev' in all_results else list(all_results.keys())[0]
            self.evaluator.compare_models(all_results[comparison_set])
        
        return all_results
    
    def run_interactive_mode(self):
        """Run interactive mode for testing sentence pairs."""
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("Enter sentence pairs to get similarity scores (type 'quit' to exit)")
        
        while True:
            print("\n" + "-"*30)
            sentence1 = input("Enter first sentence: ").strip()
            
            if sentence1.lower() == 'quit':
                break
            
            sentence2 = input("Enter second sentence: ").strip()
            
            if sentence2.lower() == 'quit':
                break
            
            # Get predictions from all models
            predictions = {}
            
            # Traditional features
            if self.models:
                # Preprocess sentences
                preprocessed_s1, preprocessed_s2 = self.preprocess_sentences([sentence1], [sentence2])
                
                features = self.feature_extractor.extract_all_features(
                    preprocessed_s1[0], preprocessed_s2[0], self.data_loader.word_counts
                )
                features = features.reshape(1, -1)
                
                trad_predictions = self.predict_with_traditional_models(features)
                for model_name, pred in trad_predictions.items():
                    predictions[model_name] = pred[0]
            
            # BERT features
            if self.use_bert:
                try:
                    bert_features = self.bert_extractor.extract_features(sentence1, sentence2)
                    bert_features = bert_features.reshape(1, -1)
                    
                    if self.use_neural:
                        neural_pred = self.predict_with_neural_model(bert_features)
                        if neural_pred is not None:
                            predictions['BERT+Neural'] = neural_pred[0]
                except Exception as e:
                    print(f"BERT prediction failed: {e}")
            
            # Print results
            print(f"\nSimilarity Scores:")
            for model_name, score in predictions.items():
                print(f"  {model_name}: {score:.3f}")
    
    def run_full_experiment(self):
        """Run the complete STS experiment."""
        print("MODULAR STS SYSTEM")
        print("="*50)
        
        # Load data
        datasets = self.load_data()
        
        if 'train' not in datasets or 'dev' not in datasets:
            print("Error: Training and development sets are required!")
            return
        
        # Extract features for training
        train_sentences1, train_sentences2, train_scores = datasets['train']
        dev_sentences1, dev_sentences2, dev_scores = datasets['dev']
        
        # Preprocess sentences
        train_sentences1, train_sentences2 = self.preprocess_sentences(train_sentences1, train_sentences2)
        dev_sentences1, dev_sentences2 = self.preprocess_sentences(dev_sentences1, dev_sentences2)
        
        if self.use_bert:
            print("Using BERT features...")
            train_features = self.extract_bert_features(train_sentences1, train_sentences2)
            dev_features = self.extract_bert_features(dev_sentences1, dev_sentences2)
        else:
            print("Using traditional features...")
            train_features = self.extract_traditional_features(train_sentences1, train_sentences2)
            dev_features = self.extract_traditional_features(dev_sentences1, dev_sentences2)
        
        # Train models
        self.train_traditional_models(train_features, train_scores, dev_features, dev_scores)
        
        if self.use_neural:
            self.train_neural_model(train_features, train_scores, dev_features, dev_scores)
        
        # Evaluate models
        results = self.evaluate_models(datasets)
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED!")
        print("="*50)
        
        return results


def main():
    """Main function to run the modular STS system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modular STS System')
    parser.add_argument('--use-bert', action='store_true', 
                       help='Use BERT embeddings (slower but potentially better)')
    parser.add_argument('--no-neural', action='store_true',
                       help='Disable neural network model')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ModularSTSSystem(
        use_bert=args.use_bert,
        use_neural=not args.no_neural
    )
    
    if args.interactive:
        system.run_interactive_mode()
    else:
        system.run_full_experiment()


if __name__ == "__main__":
    main() 