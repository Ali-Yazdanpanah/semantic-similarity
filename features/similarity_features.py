"""
Similarity Features for STS
"""

import numpy as np
import difflib
import string
from collections import defaultdict
from typing import List, Dict, Optional

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn, wordnet_ic

# Gensim imports
from gensim.models import KeyedVectors


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