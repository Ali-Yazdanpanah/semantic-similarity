"""
BERT Embeddings for STS
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import time


class BERTSimilarity:
    """BERT-based sentence similarity."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        
        # Set model to evaluation mode
        self.model.eval()
        
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get BERT embedding for a single sentence."""
        inputs = self.tokenizer(
            sentence, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.cpu().numpy().flatten()
    
    def get_sentence_pair_embedding(self, sentence1: str, sentence2: str) -> np.ndarray:
        """Get BERT embedding for a sentence pair."""
        inputs = self.tokenizer(
            sentence1, 
            sentence2,
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.cpu().numpy().flatten()
    
    def compute_similarity(self, sentence1: str, sentence2: str) -> float:
        """Compute cosine similarity between two sentences."""
        emb1 = self.get_sentence_embedding(sentence1)
        emb2 = self.get_sentence_embedding(sentence2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def get_batch_embeddings(self, sentences: List[str], batch_size: int = 8) -> np.ndarray:
        """Get embeddings for a batch of sentences."""
        embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentence_pair_features(self, sentence1: str, sentence2: str) -> np.ndarray:
        """Get features for sentence pair similarity."""
        # Get individual embeddings
        emb1 = self.get_sentence_embedding(sentence1)
        emb2 = self.get_sentence_embedding(sentence2)
        
        # Get pair embedding
        pair_emb = self.get_sentence_pair_embedding(sentence1, sentence2)
        
        # Compute similarity features
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        manhattan_dist = np.sum(np.abs(emb1 - emb2))
        
        # Combine features
        features = np.concatenate([
            emb1,  # Sentence 1 embedding
            emb2,  # Sentence 2 embedding
            pair_emb,  # Pair embedding
            [cosine_sim],  # Cosine similarity
            [euclidean_dist],  # Euclidean distance
            [manhattan_dist]  # Manhattan distance
        ])
        
        return features


class BERTFeatureExtractor:
    """Extract BERT-based features for STS."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.bert = BERTSimilarity(model_name)
        
    def extract_features(self, sentence1: str, sentence2: str) -> np.ndarray:
        """Extract BERT features for sentence pair."""
        try:
            features = self.bert.get_sentence_pair_features(sentence1, sentence2)
            return features
        except Exception as e:
            print(f"Error extracting BERT features: {e}")
            # Return zero features if BERT fails
            return np.zeros(768 * 3 + 3)  # 3 embeddings + 3 similarity measures
    
    def extract_batch_features(self, sentence_pairs: List[Tuple[str, str]], 
                             batch_size: int = 8, verbose: bool = True) -> np.ndarray:
        """Extract BERT features for multiple sentence pairs."""
        features = []
        total_pairs = len(sentence_pairs)
        
        start_time = time.time()
        for i, (s1, s2) in enumerate(sentence_pairs):
            feature = self.extract_features(s1, s2)
            features.append(feature)
            
            if verbose and (i + 1) % max(1, total_pairs // 10) == 0:
                progress = (i + 1) * 100.0 / total_pairs
                elapsed = time.time() - start_time
                print(f"BERT features: {progress:.2f}% finished ({elapsed:.2f}s)")
        
        return np.array(features) 