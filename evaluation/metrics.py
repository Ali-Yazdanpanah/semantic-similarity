"""
Evaluation Metrics for STS
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict


class STSEvaluator:
    """Evaluator for STS predictions."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_pearson_correlation(self, true_scores: List[float], 
                                    predicted_scores: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Pearson correlation coefficient
        """
        correlation, _ = pearsonr(true_scores, predicted_scores)
        return correlation
    
    def calculate_spearman_correlation(self, true_scores: List[float], 
                                     predicted_scores: List[float]) -> float:
        """
        Calculate Spearman rank correlation coefficient.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Spearman correlation coefficient
        """
        correlation, _ = spearmanr(true_scores, predicted_scores)
        return correlation
    
    def calculate_mse(self, true_scores: List[float], 
                     predicted_scores: List[float]) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Mean Squared Error
        """
        return mean_squared_error(true_scores, predicted_scores)
    
    def calculate_rmse(self, true_scores: List[float], 
                      predicted_scores: List[float]) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Root Mean Squared Error
        """
        mse = mean_squared_error(true_scores, predicted_scores)
        return np.sqrt(mse)
    
    def calculate_mae(self, true_scores: List[float], 
                     predicted_scores: List[float]) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Mean Absolute Error
        """
        return mean_absolute_error(true_scores, predicted_scores)
    
    def calculate_all_metrics(self, true_scores: List[float], 
                            predicted_scores: List[float]) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'pearson': self.calculate_pearson_correlation(true_scores, predicted_scores),
            'spearman': self.calculate_spearman_correlation(true_scores, predicted_scores),
            'mse': self.calculate_mse(true_scores, predicted_scores),
            'rmse': self.calculate_rmse(true_scores, predicted_scores),
            'mae': self.calculate_mae(true_scores, predicted_scores)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], dataset_name: str = "Dataset"):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            dataset_name: Name of the dataset being evaluated
        """
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS - {dataset_name.upper()}")
        print(f"{'='*50}")
        
        print(f"Pearson Correlation:  {metrics['pearson']:.4f}")
        print(f"Spearman Correlation: {metrics['spearman']:.4f}")
        print(f"Mean Squared Error:   {metrics['mse']:.4f}")
        print(f"Root MSE:             {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error:  {metrics['mae']:.4f}")
        
        print(f"{'='*50}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]):
        """
        Compare multiple models and print results.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        # Print header
        print(f"{'Model':<20} {'Pearson':<10} {'Spearman':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 80)
        
        # Print results for each model
        for model_name, metrics in model_results.items():
            print(f"{model_name:<20} {metrics['pearson']:<10.4f} "
                  f"{metrics['spearman']:<10.4f} {metrics['rmse']:<10.4f} "
                  f"{metrics['mae']:<10.4f}")
        
        print(f"{'='*80}")
    
    def analyze_predictions(self, true_scores: List[float], 
                          predicted_scores: List[float], 
                          sentences1: List[str], 
                          sentences2: List[str],
                          num_examples: int = 5) -> Dict[str, List[Tuple]]:
        """
        Analyze predictions and find examples of good and bad predictions.
        
        Args:
            true_scores: Ground truth similarity scores
            predicted_scores: Predicted similarity scores
            sentences1: First sentences
            sentences2: Second sentences
            num_examples: Number of examples to return for each category
            
        Returns:
            Dictionary containing examples of good and bad predictions
        """
        # Calculate prediction errors
        errors = [abs(true - pred) for true, pred in zip(true_scores, predicted_scores)]
        
        # Find indices of best and worst predictions
        best_indices = np.argsort(errors)[:num_examples]
        worst_indices = np.argsort(errors)[-num_examples:][::-1]
        
        # Collect examples
        best_examples = []
        worst_examples = []
        
        for idx in best_indices:
            best_examples.append((
                sentences1[idx], 
                sentences2[idx], 
                true_scores[idx], 
                predicted_scores[idx], 
                errors[idx]
            ))
        
        for idx in worst_indices:
            worst_examples.append((
                sentences1[idx], 
                sentences2[idx], 
                true_scores[idx], 
                predicted_scores[idx], 
                errors[idx]
            ))
        
        return {
            'best_predictions': best_examples,
            'worst_predictions': worst_examples
        }
    
    def print_prediction_analysis(self, analysis: Dict[str, List[Tuple]], 
                                num_examples: int = 3):
        """
        Print prediction analysis results.
        
        Args:
            analysis: Dictionary containing analysis results
            num_examples: Number of examples to print
        """
        print(f"\n{'='*80}")
        print("PREDICTION ANALYSIS")
        print(f"{'='*80}")
        
        # Print best predictions
        print(f"\nBEST PREDICTIONS (Error < {analysis['best_predictions'][0][4]:.3f}):")
        for i, (s1, s2, true, pred, error) in enumerate(analysis['best_predictions'][:num_examples], 1):
            print(f"  {i}. True: {true:.2f}, Pred: {pred:.2f}, Error: {error:.3f}")
            print(f"     S1: {s1}")
            print(f"     S2: {s2}")
            print()
        
        # Print worst predictions
        print(f"\nWORST PREDICTIONS (Error > {analysis['worst_predictions'][0][4]:.3f}):")
        for i, (s1, s2, true, pred, error) in enumerate(analysis['worst_predictions'][:num_examples], 1):
            print(f"  {i}. True: {true:.2f}, Pred: {pred:.2f}, Error: {error:.3f}")
            print(f"     S1: {s1}")
            print(f"     S2: {s2}")
            print()
        
        print(f"{'='*80}")
    
    def calculate_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate distribution statistics for scores.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary containing distribution statistics
        """
        scores_array = np.array(scores)
        
        distribution = {
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'min': np.min(scores_array),
            'max': np.max(scores_array),
            'median': np.median(scores_array),
            'q25': np.percentile(scores_array, 25),
            'q75': np.percentile(scores_array, 75)
        }
        
        return distribution
    
    def print_score_distribution(self, distribution: Dict[str, float], 
                               score_type: str = "Scores"):
        """
        Print score distribution statistics.
        
        Args:
            distribution: Dictionary containing distribution statistics
            score_type: Type of scores being analyzed
        """
        print(f"\n{score_type.upper()} DISTRIBUTION:")
        print(f"  Mean:   {distribution['mean']:.3f}")
        print(f"  Std:    {distribution['std']:.3f}")
        print(f"  Min:    {distribution['min']:.3f}")
        print(f"  Max:    {distribution['max']:.3f}")
        print(f"  Median: {distribution['median']:.3f}")
        print(f"  Q25:    {distribution['q25']:.3f}")
        print(f"  Q75:    {distribution['q75']:.3f}") 