#!/usr/bin/env python3
"""
Semantic Textual Similarity (STS) System with PyTorch

This module implements various similarity measures for sentence pairs and uses them
as features for a Support Vector Regression model to predict similarity scores.

Author: Ali Yazdanpanah
Class: Text Mining
"""

import os
import glob
import time
import string
import difflib
import numpy as np
import sklearn
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# NLTK imports
import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, wordnet_ic

# Machine learning imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Gensim imports
import gensim
from gensim.models import KeyedVectors

# Download required NLTK data
try:
    nltk.download('wordnet_ic', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")


class STSDataset(Dataset):
    """PyTorch Dataset for STS data."""
    
    def __init__(self, sent1_list: List[str], sent2_list: List[str], 
                 scores: Optional[List[float]] = None):
        self.sent1_list = sent1_list
        self.sent2_list = sent2_list
        self.scores = scores if scores is not None else [0.0] * len(sent1_list)
    
    def __len__(self):
        return len(self.sent1_list)
    
    def __getitem__(self, idx):
        return {
            'sentence1': self.sent1_list[idx],
            'sentence2': self.sent2_list[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32)
        }


class TextPreprocessor:
    """Handles text preprocessing including tokenization and lemmatization."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TweetTokenizer()
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> set:
        """Load a comprehensive list of stop words."""
        return {
            "a", "a's", "able", "about", "above", "according", "accordingly", "across",
            "actually", "after", "afterwards", "again", "against", "ain't", "all",
            "allow", "allows", "almost", "alone", "along", "already", "also",
            "although", "always", "am", "among", "amongst", "an", "and", "another",
            "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways",
            "anywhere", "apart", "appear", "appreciate", "appropriate", "are",
            "aren't", "around", "as", "aside", "ask", "asking", "associated", "at",
            "available", "away", "awfully", "b", "be", "became", "because", "become",
            "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
            "believe", "below", "beside", "besides", "best", "better", "between",
            "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came",
            "can", "can't", "cannot", "cant", "cause", "causes", "certain",
            "certainly", "changes", "clearly", "co", "com", "come", "comes",
            "concerning", "consequently", "consider", "considering", "contain",
            "containing", "contains", "corresponding", "could", "couldn't", "course",
            "currently", "d", "definitely", "described", "despite", "did", "didn't",
            "different", "do", "does", "doesn't", "doing", "don't", "done", "down",
            "downwards", "during", "e", "each", "edu", "eg", "eight", "either",
            "else", "elsewhere", "enough", "entirely", "especially", "et", "etc",
            "even", "ever", "every", "everybody", "everyone", "everything",
            "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
            "fifth", "first", "five", "followed", "following", "follows", "for",
            "former", "formerly", "forth", "four", "from", "further", "furthermore",
            "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going",
            "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens",
            "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's",
            "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby",
            "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his",
            "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll",
            "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch",
            "inc", "indeed", "indicate", "indicated", "indicates", "inner",
            "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd",
            "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps",
            "kept", "know", "knows", "known", "l", "last", "lately", "later",
            "latter", "latterly", "least", "less", "lest", "let", "let's", "like",
            "liked", "likely", "little", "look", "looking", "looks", "ltd", "m",
            "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely",
            "might", "more", "moreover", "most", "mostly", "much", "must", "my",
            "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary",
            "need", "needs", "neither", "never", "nevertheless", "new", "next",
            "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
            "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of",
            "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
            "only", "onto", "or", "other", "others", "otherwise", "ought", "our",
            "ours", "ourselves", "out", "outside", "over", "overall", "own", "p",
            "particular", "particularly", "per", "perhaps", "placed", "please",
            "plus", "possible", "presumably", "probably", "provides", "q", "que",
            "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably",
            "regarding", "regardless", "regards", "relatively", "respectively",
            "right", "s", "said", "same", "saw", "say", "saying", "says", "second",
            "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems",
            "seen", "self", "selves", "sensible", "sent", "serious", "seriously",
            "seven", "several", "shall", "she", "should", "shouldn't", "since",
            "six", "so", "some", "somebody", "somehow", "someone", "something",
            "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry",
            "specified", "specify", "specifying", "still", "sub", "such", "sup",
            "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than",
            "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their",
            "theirs", "them", "themselves", "then", "thence", "there", "there's",
            "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
            "these", "they", "they'd", "they'll", "they're", "they've", "think",
            "third", "this", "thorough", "thoroughly", "those", "though", "three",
            "through", "throughout", "thru", "thus", "to", "together", "too",
            "took", "toward", "towards", "tried", "tries", "truly", "try",
            "trying", "twice", "two", "u", "un", "under", "unfortunately",
            "unless", "unlikely", "until", "unto", "up", "upon", "us", "use",
            "used", "useful", "uses", "using", "usually", "uucp", "v", "value",
            "various", "very", "via", "viz", "vs", "w", "want", "wants", "was",
            "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome",
            "well", "went", "were", "weren't", "what", "what's", "whatever",
            "when", "whence", "whenever", "where", "where's", "whereafter",
            "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
            "which", "while", "whither", "who", "who's", "whoever", "whole",
            "whom", "whose", "why", "will", "willing", "wish", "with", "within",
            "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes",
            "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves", "z", "zero"
        }
    
    def normalize_corpus(self, corpus: List[List[str]]) -> List[List[str]]:
        """
        Normalize a corpus of tokenized sentences.
        
        Args:
            corpus: List of tokenized sentences
            
        Returns:
            Normalized corpus with lemmatization and stop word removal
        """
        normalized_corpus = []
        for sentence in corpus:
            normalized_sentence = []
            for word in sentence:
                if word.isdecimal() or word.isdigit():
                    normalized_sentence.append('NUM')
                elif word in string.punctuation:
                    pass
                elif word in self.stop_words:
                    pass
                else:
                    normalized_sentence.append(self.lemmatizer.lemmatize(word.lower()))
            normalized_corpus.append(normalized_sentence)
        return normalized_corpus
    
    def preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """
        Preprocess a list of sentences.
        
        Args:
            sentences: List of raw sentences
            
        Returns:
            List of preprocessed sentences
        """
        # Tokenize
        tokenized = [self.tokenizer.tokenize(s) for s in sentences]
        # Normalize
        normalized = self.normalize_corpus(tokenized)
        # Reconstruct sentences
        processed_sentences = []
        for sentence in normalized:
            processed_sentences.append(" ".join(sentence))
        return processed_sentences


class SimilarityFeatures:
    """Implements various similarity measures for sentence pairs."""
    
    def __init__(self, word_vectors: Optional[KeyedVectors] = None):
        """
        Initialize similarity features calculator.
        
        Args:
            word_vectors: Pre-trained word vectors (optional)
        """
        self.word_vectors = word_vectors
        self.brown_ic = None
        try:
            self.brown_ic = wordnet_ic.ic('ic-brown.dat')
        except:
            print("Warning: Brown corpus information content not available.")
    
    def simple_baseline_similarity(self, s1: str, s2: str, 
                                 word_counts: Optional[Dict[str, int]] = None) -> float:
        """
        Simple baseline similarity using word overlap.
        
        Args:
            s1: First sentence
            s2: Second sentence
            word_counts: Optional word frequency counts for TF-IDF weighting
            
        Returns:
            Similarity score between 0 and 5
        """
        def get_embedding(sentence: str) -> Dict[str, int]:
            embedding = defaultdict(int)
            for word in sentence.split():
                word = word.strip('?.,')
                embedding[word] += 1
            return embedding
        
        first_embedding = get_embedding(s1)
        second_embedding = get_embedding(s2)
        
        embedding1 = []
        embedding2 = []
        
        if word_counts:
            for word in first_embedding:
                embedding1.append(first_embedding[word] * 1.0 / (word_counts.get(word, 0) + 0.001))
                embedding2.append(second_embedding.get(word, 0) * 1.0 / (word_counts.get(word, 0) + 0.001))
        else:
            for word in first_embedding:
                embedding1.append(first_embedding[word])
                embedding2.append(second_embedding.get(word, 0))
        
        if sum(embedding2) == 0:
            return 0.0
        
        sm = difflib.SequenceMatcher(None, embedding1, embedding2)
        return sm.ratio() * 5
    
    def word_alignment_similarity(self, s1: str, s2: str) -> float:
        """
        Word alignment similarity using WordNet.
        
        Args:
            s1: First sentence
            s2: Second sentence
            
        Returns:
            Similarity score
        """
        def penn_to_wn(tag: str) -> Optional[str]:
            """Convert Penn Treebank tag to WordNet tag."""
            if tag.startswith('N'): return 'n'
            if tag.startswith('V'): return 'v'
            if tag.startswith('J'): return 'a'
            if tag.startswith('R'): return 'r'
            return None
        
        def tagged_to_synset(word: str, tag: str):
            """Get synset for tagged word."""
            wn_tag = penn_to_wn(tag)
            if wn_tag is None:
                return None
            try:
                return wn.synsets(word, wn_tag)[0]
            except:
                return None
        
        # Tokenize and tag
        sentence1 = pos_tag(word_tokenize(s1))
        sentence2 = pos_tag(word_tokenize(s2))
        
        # Get synsets
        synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
        
        # Filter out None values
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]
        
        score, count = 0.0, 0
        
        # Calculate similarity
        for synset in synsets1:
            similarities = [synset.path_similarity(ss) for ss in synsets2]
            similarities = [s for s in similarities if s is not None]
            if similarities:
                best_score = max(similarities)
                score += best_score
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def information_content_similarity(self, s1: str, s2: str) -> float:
        """
        Information content similarity using WordNet and Brown corpus.
        
        Args:
            s1: First sentence
            s2: Second sentence
            
        Returns:
            Similarity score
        """
        if self.brown_ic is None:
            return 0.0
        
        def penn_to_wn(tag: str) -> Optional[str]:
            """Convert Penn Treebank tag to WordNet tag."""
            if tag.startswith('N'): return 'n'
            if tag.startswith('V'): return 'v'
            if tag.startswith('J'): return 'a'
            if tag.startswith('R'): return 'r'
            return None
        
        def tagged_to_synset(word: str, tag: str):
            """Get synset for tagged word."""
            wn_tag = penn_to_wn(tag)
            if wn_tag is None:
                return None
            try:
                return wn.synsets(word, wn_tag)[0]
            except:
                return None
        
        # Tokenize and tag
        sentence1 = pos_tag(word_tokenize(s1))
        sentence2 = pos_tag(word_tokenize(s2))
        
        # Get synsets
        synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
        
        # Filter out None values
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]
        
        score, count = 0.0, 0
        
        # Calculate Resnik similarity
        for synset in synsets1:
            similarities = []
            for ss in synsets2:
                try:
                    similarities.append(synset.res_similarity(ss, self.brown_ic))
                except:
                    continue
            if similarities:
                best_score = max(similarities)
                score += best_score
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def unigram_overlap(self, s1: str, s2: str) -> float:
        """
        Calculate unigram overlap between sentences.
        
        Args:
            s1: First sentence
            s2: Second sentence
            
        Returns:
            Overlap score
        """
        words1 = s1.strip().split()
        words2 = s2.strip().split()
        
        overlap_count = 0
        for word1 in words1:
            overlap_count += words2.count(word1)
        
        return 2 * overlap_count / (len(words1) + len(words2) + 0.0)
    
    def absolute_difference_features(self, s1: str, s2: str) -> List[float]:
        """
        Calculate absolute difference features for different POS types.
        
        Args:
            s1: First sentence
            s2: Second sentence
            
        Returns:
            List of 5 features: [all_tokens, adjectives, adverbs, nouns, verbs]
        """
        tokens1, tokens2 = word_tokenize(s1), word_tokenize(s2)
        pos1, pos2 = pos_tag(tokens1), pos_tag(tokens2)
        
        features = []
        
        # All tokens
        t1 = abs(len(tokens1) - len(tokens2)) / float(len(tokens1) + len(tokens2))
        features.append(t1)
        
        # Adjectives
        cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2) if (cnt1 + cnt2) > 0 else 0
        features.append(t2)
        
        # Adverbs
        cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2) if (cnt1 + cnt2) > 0 else 0
        features.append(t3)
        
        # Nouns
        cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2) if (cnt1 + cnt2) > 0 else 0
        features.append(t4)
        
        # Verbs
        cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2) if (cnt1 + cnt2) > 0 else 0
        features.append(t5)
        
        return features
    
    def min_max_ratio_features(self, s1: str, s2: str) -> List[float]:
        """
        Calculate min-to-max ratio features for different POS types.
        
        Args:
            s1: First sentence
            s2: Second sentence
            
        Returns:
            List of 5 features: [all_tokens, adjectives, adverbs, nouns, verbs]
        """
        shorter = 1 if len(s1) <= len(s2) else 2
        
        tokens1, tokens2 = word_tokenize(s1), word_tokenize(s2)
        pos1, pos2 = pos_tag(tokens1), pos_tag(tokens2)
        
        features = []
        
        # All tokens
        t1 = (len(tokens1) + 0.001) / (len(tokens2) + 0.001)
        features.append(t1)
        
        # Adjectives
        cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
        t2 = (cnt1 + 0.001) / (cnt2 + 0.001) if (cnt1 + cnt2) > 0 else 0
        features.append(t2)
        
        # Adverbs
        cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
        t3 = (cnt1 + 0.001) / (cnt2 + 0.001) if (cnt1 + cnt2) > 0 else 0
        features.append(t3)
        
        # Nouns
        cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
        t4 = (cnt1 + 0.001) / (cnt2 + 0.001) if (cnt1 + cnt2) > 0 else 0
        features.append(t4)
        
        # Verbs
        cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
        cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
        t5 = (cnt1 + 0.001) / (cnt2 + 0.001) if (cnt1 + cnt2) > 0 else 0
        features.append(t5)
        
        # Adjust if second sentence is shorter
        if shorter == 2:
            features = [1 / (f + 0.001) for f in features]
        
        return features
    
    def extract_all_features(self, s1: str, s2: str, 
                           word_counts: Optional[Dict[str, int]] = None) -> List[float]:
        """
        Extract all similarity features for a sentence pair.
        
        Args:
            s1: First sentence
            s2: Second sentence
            word_counts: Optional word frequency counts
            
        Returns:
            List of all features
        """
        features = [
            self.simple_baseline_similarity(s1, s2, word_counts),
            self.word_alignment_similarity(s1, s2),
            self.information_content_similarity(s1, s2),
            self.unigram_overlap(s1, s2)
        ]
        
        # Add absolute difference features
        features.extend(self.absolute_difference_features(s1, s2))
        
        # Add min-max ratio features
        features.extend(self.min_max_ratio_features(s1, s2))
        
        return features


class STSModel:
    """Main STS model that combines similarity features with SVR."""
    
    def __init__(self, word_vectors_path: Optional[str] = None, use_pytorch: bool = True):
        """
        Initialize STS model.
        
        Args:
            word_vectors_path: Path to pre-trained word vectors (optional)
            use_pytorch: Whether to use PyTorch for feature extraction
        """
        self.preprocessor = TextPreprocessor()
        self.similarity_features = SimilarityFeatures()
        self.scaler = StandardScaler()
        self.model = SVR(kernel='linear')
        self.word_counts = defaultdict(int)
        self.use_pytorch = use_pytorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load word vectors if provided
        if word_vectors_path and os.path.exists(word_vectors_path):
            try:
                self.similarity_features.word_vectors = KeyedVectors.load_word2vec_format(
                    word_vectors_path, binary=True
                )
                print(f"Loaded word vectors from {word_vectors_path}")
            except Exception as e:
                print(f"Warning: Could not load word vectors: {e}")
    
    def preprocess_data(self, train_file: str, dev_file: str, test_file: str) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[float], List[str], List[str], List[float]]:
        """
        Preprocess training, development, and test data.
        
        Args:
            train_file: Path to training file
            dev_file: Path to development file
            test_file: Path to test file
            
        Returns:
            Tuple of (train_sent1, train_sent2, train_scores, dev_sent1, dev_sent2, dev_scores, test_sent1, test_sent2, test_scores)
        """
        # Load training data
        train_sent1, train_sent2, train_scores = [], [], []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    train_sent1.append(parts[1])
                    train_sent2.append(parts[2])
                    train_scores.append(float(parts[3]))
        
        # Load development data
        dev_sent1, dev_sent2, dev_scores = [], [], []
        with open(dev_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    dev_sent1.append(parts[1])
                    dev_sent2.append(parts[2])
                    dev_scores.append(float(parts[3]))
        
        # Load test data
        test_sent1, test_sent2, test_scores = [], [], []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    test_sent1.append(parts[1])
                    test_sent2.append(parts[2])
                    test_scores.append(float(parts[3]))
        
        # Preprocess sentences
        print("Preprocessing training sentences...")
        train_sent1 = self.preprocessor.preprocess_sentences(train_sent1)
        train_sent2 = self.preprocessor.preprocess_sentences(train_sent2)
        
        print("Preprocessing development sentences...")
        dev_sent1 = self.preprocessor.preprocess_sentences(dev_sent1)
        dev_sent2 = self.preprocessor.preprocess_sentences(dev_sent2)
        
        print("Preprocessing test sentences...")
        test_sent1 = self.preprocessor.preprocess_sentences(test_sent1)
        test_sent2 = self.preprocessor.preprocess_sentences(test_sent2)
        
        # Build word frequency counts
        all_sentences = train_sent1 + train_sent2 + dev_sent1 + dev_sent2 + test_sent1 + test_sent2
        self._build_word_counts(all_sentences)
        
        return train_sent1, train_sent2, train_scores, dev_sent1, dev_sent2, dev_scores, test_sent1, test_sent2, test_scores
    
    def _build_word_counts(self, sentences: List[str]):
        """Build word frequency counts for TF-IDF weighting."""
        for sentence in sentences:
            for word in sentence.split():
                word = word.strip('?.,')
                self.word_counts[word] += 1
    
    def extract_features(self, sent1_list: List[str], sent2_list: List[str], 
                        verbose: bool = True) -> np.ndarray:
        """
        Extract features for all sentence pairs.
        
        Args:
            sent1_list: List of first sentences
            sent2_list: List of second sentences
            verbose: Whether to print progress
            
        Returns:
            Feature matrix
        """
        features = []
        n_pairs = len(sent1_list)
        
        start_time = time.time()
        for i, (s1, s2) in enumerate(zip(sent1_list, sent2_list)):
            feature_vector = self.similarity_features.extract_all_features(
                s1, s2, self.word_counts
            )
            features.append(feature_vector)
            
            if verbose and (i + 1) % max(1, n_pairs // 10) == 0:
                progress = (i + 1) * 100.0 / n_pairs
                elapsed = time.time() - start_time
                print(f"{progress:.2f}% finished ({elapsed:.2f}s)")
        
        return np.array(features)
    
    def train(self, train_sent1: List[str], train_sent2: List[str], 
              train_scores: List[float], dev_sent1: List[str], dev_sent2: List[str], 
              dev_scores: List[float], verbose: bool = True):
        """
        Train the STS model with validation.
        
        Args:
            train_sent1: List of first training sentences
            train_sent2: List of second training sentences
            train_scores: List of training scores
            dev_sent1: List of first development sentences
            dev_sent2: List of second development sentences
            dev_scores: List of development scores
            verbose: Whether to print progress
        """
        print("Extracting training features...")
        train_features = self.extract_features(train_sent1, train_sent2, verbose)
        
        print("Extracting development features...")
        dev_features = self.extract_features(dev_sent1, dev_sent2, verbose)
        
        print("Scaling features...")
        train_features_scaled = self.scaler.fit_transform(train_features)
        dev_features_scaled = self.scaler.transform(dev_features)
        
        print("Training SVR model...")
        self.model.fit(train_features_scaled, train_scores)
        
        # Evaluate on development set
        dev_predictions = self.model.predict(dev_features_scaled)
        dev_correlation = pearsonr(dev_scores, dev_predictions)[0]
        
        print(f"Development set Pearson correlation: {dev_correlation:.4f}")
        print("Training completed!")
    
    def predict(self, test_sent1: List[str], test_sent2: List[str], 
                verbose: bool = True) -> List[float]:
        """
        Predict similarity scores for test sentence pairs.
        
        Args:
            test_sent1: List of first test sentences
            test_sent2: List of second test sentences
            verbose: Whether to print progress
            
        Returns:
            List of predicted scores
        """
        print("Extracting test features...")
        test_features = self.extract_features(test_sent1, test_sent2, verbose)
        
        print("Scaling features...")
        test_features_scaled = self.scaler.transform(test_features)
        
        print("Making predictions...")
        predictions = self.model.predict(test_features_scaled)
        
        # Clip predictions to [0, 5] range
        predictions = [max(0, min(5, p)) for p in predictions]
        
        return predictions
    
    def evaluate(self, test_file: str, predictions: List[float]) -> float:
        """
        Evaluate predictions using Pearson correlation.
        
        Args:
            test_file: Path to test file with gold scores
            predictions: List of predicted scores
            
        Returns:
            Pearson correlation score
        """
        gold_scores = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    gold_scores.append(float(parts[3]))
        
        if len(gold_scores) != len(predictions):
            print(f"Warning: Mismatch in number of predictions ({len(predictions)}) "
                  f"and gold scores ({len(gold_scores)})")
            min_len = min(len(predictions), len(gold_scores))
            predictions = predictions[:min_len]
            gold_scores = gold_scores[:min_len]
        
        correlation = pearsonr(gold_scores, predictions)[0]
        return correlation


def preprocess_input_files():
    """Preprocess input files by removing indices."""
    for filename in glob.glob('*.txt'):
        if filename in ['test.txt', 'train.txt', 'dev.txt']:
            output_filename = f"new_{filename}"
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        f.write(f"{parts[1]}\t{parts[2]}\t{parts[3]}\n")
            
            print(f"Created {output_filename}")


def main():
    """Main function to run the STS system."""
    print("Semantic Textual Similarity (STS) System with PyTorch")
    print("=" * 60)
    
    # Check for input files
    if not all(os.path.exists(f) for f in ['train.txt', 'dev.txt', 'test.txt']):
        print("Error: train.txt, dev.txt, and test.txt files not found!")
        print("Please run: python download_sts_dataset.py")
        return
    
    # Preprocess input files
    print("Preprocessing input files...")
    preprocess_input_files()
    
    # Initialize model
    print("Initializing STS model...")
    model = STSModel(use_pytorch=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_sent1, train_sent2, train_scores, dev_sent1, dev_sent2, dev_scores, test_sent1, test_sent2, test_scores = model.preprocess_data(
        'new_train.txt', 'new_dev.txt', 'new_test.txt'
    )
    
    # Train model with validation
    print(f"Training on {len(train_sent1)} sentence pairs with {len(dev_sent1)} validation pairs...")
    start_time = time.time()
    model.train(train_sent1, train_sent2, train_scores, dev_sent1, dev_sent2, dev_scores)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions on test set
    print(f"Making predictions on {len(test_sent1)} sentence pairs...")
    predictions = model.predict(test_sent1, test_sent2)
    
    # Save predictions
    with open('results.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print("Predictions saved to results.txt")
    
    # Evaluate
    try:
        correlation = model.evaluate('new_test.txt', predictions)
        print(f"Test set Pearson correlation: {correlation:.4f}")
    except Exception as e:
        print(f"Could not evaluate: {e}")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 