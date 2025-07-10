"""
Dataset Loader for STS
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class STSDatasetLoader:
    """Load and preprocess STS datasets."""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = data_dir
        self.word_counts = None
        
    def load_dataset(self, filename: str) -> Tuple[List[str], List[str], List[float]]:
        """
        Load dataset from file.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Tuple of (sentences1, sentences2, scores)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        sentences1, sentences2, scores = [], [], []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 3:
                    print(f"Warning: Skipping line {line_num} with insufficient fields (found {len(parts)}, need 3)")
                    continue
                
                try:
                    s1, s2, score = parts
                    score = float(score)
                    sentences1.append(s1)
                    sentences2.append(s2)
                    scores.append(score)
                except ValueError:
                    print(f"Warning: Skipping line {line_num} with invalid score: {parts[2]}")
                    continue
        
        print(f"Loaded {len(sentences1)} pairs from {filename}")
        return sentences1, sentences2, scores
    
    def build_word_counts(self, sentences: List[str]) -> Dict[str, int]:
        """
        Build word frequency counts from sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dictionary of word counts
        """
        word_counts = defaultdict(int)
        
        for sentence in sentences:
            for word in sentence.split():
                word = word.strip('?.,')
                word_counts[word] += 1
        
        return dict(word_counts)
    
    def load_all_datasets(self) -> Dict[str, Tuple[List[str], List[str], List[float]]]:
        """
        Load all available datasets (train, dev, test).
        
        Returns:
            Dictionary mapping dataset names to (sentences1, sentences2, scores)
        """
        datasets = {}
        
        # Try to load each dataset
        for split in ['train', 'dev', 'test']:
            filename = f"{split}.txt"
            try:
                sentences1, sentences2, scores = self.load_dataset(filename)
                datasets[split] = (sentences1, sentences2, scores)
            except FileNotFoundError:
                print(f"Warning: {filename} not found, skipping...")
        
        # Build word counts from training data if available
        if 'train' in datasets:
            all_train_sentences = datasets['train'][0] + datasets['train'][1]
            self.word_counts = self.build_word_counts(all_train_sentences)
            print(f"Built word counts from {len(all_train_sentences)} training sentences")
        
        return datasets
    
    def get_dataset_statistics(self, datasets: Dict[str, Tuple[List[str], List[str], List[float]]]) -> Dict[str, Dict]:
        """
        Get statistics for all datasets.
        
        Args:
            datasets: Dictionary of datasets
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        for split_name, (sentences1, sentences2, scores) in datasets.items():
            if not scores:  # Skip if no scores (e.g., test set)
                continue
                
            stats[split_name] = {
                'num_pairs': len(sentences1),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'avg_length_s1': np.mean([len(s.split()) for s in sentences1]),
                'avg_length_s2': np.mean([len(s.split()) for s in sentences2])
            }
        
        return stats
    
    def print_dataset_info(self, datasets: Dict[str, Tuple[List[str], List[str], List[float]]]):
        """
        Print information about loaded datasets.
        
        Args:
            datasets: Dictionary of datasets
        """
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        stats = self.get_dataset_statistics(datasets)
        
        for split_name, (sentences1, sentences2, scores) in datasets.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Number of pairs: {len(sentences1)}")
            
            if scores:
                print(f"  Score range: {min(scores):.2f} - {max(scores):.2f}")
                print(f"  Average score: {np.mean(scores):.2f}")
                print(f"  Score std dev: {np.std(scores):.2f}")
            else:
                print(f"  No scores available (test set)")
            
            print(f"  Average sentence 1 length: {np.mean([len(s.split()) for s in sentences1]):.1f} words")
            print(f"  Average sentence 2 length: {np.mean([len(s.split()) for s in sentences2]):.1f} words")
        
        if self.word_counts:
            print(f"\nVocabulary size: {len(self.word_counts)} unique words")
        
        print("="*50)
    
    def get_sample_pairs(self, datasets: Dict[str, Tuple[List[str], List[str], List[float]]], 
                        num_samples: int = 5) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Get sample sentence pairs from each dataset.
        
        Args:
            datasets: Dictionary of datasets
            num_samples: Number of samples to return per dataset
            
        Returns:
            Dictionary of sample pairs
        """
        samples = {}
        
        for split_name, (sentences1, sentences2, scores) in datasets.items():
            if not scores:
                continue
                
            # Get random samples
            indices = np.random.choice(len(sentences1), 
                                     size=min(num_samples, len(sentences1)), 
                                     replace=False)
            
            split_samples = []
            for idx in indices:
                split_samples.append((sentences1[idx], sentences2[idx], scores[idx]))
            
            samples[split_name] = split_samples
        
        return samples
    
    def print_sample_pairs(self, datasets: Dict[str, Tuple[List[str], List[str], List[float]]], 
                          num_samples: int = 3):
        """
        Print sample sentence pairs from each dataset.
        
        Args:
            datasets: Dictionary of datasets
            num_samples: Number of samples to show per dataset
        """
        samples = self.get_sample_pairs(datasets, num_samples)
        
        print("\n" + "="*50)
        print("SAMPLE SENTENCE PAIRS")
        print("="*50)
        
        for split_name, split_samples in samples.items():
            print(f"\n{split_name.upper()} SET SAMPLES:")
            for i, (s1, s2, score) in enumerate(split_samples, 1):
                print(f"  {i}. Score: {score:.2f}")
                print(f"     S1: {s1}")
                print(f"     S2: {s2}")
                print()
        
        print("="*50) 