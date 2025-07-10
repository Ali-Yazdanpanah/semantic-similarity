"""
Text Preprocessor for STS
"""

import string
from typing import List, Set
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """Handles text preprocessing including tokenization and lemmatization."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TweetTokenizer()
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> Set[str]:
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
        
        # Join back into sentences
        preprocessed = [' '.join(sentence) for sentence in normalized]
        
        return preprocessed
    
    def tokenize_sentence(self, sentence: str) -> List[str]:
        """
        Tokenize a single sentence.
        
        Args:
            sentence: Raw sentence
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(sentence)
    
    def lemmatize_word(self, word: str, pos: str = None) -> str:
        """
        Lemmatize a single word.
        
        Args:
            word: Word to lemmatize
            pos: Part of speech tag (optional)
            
        Returns:
            Lemmatized word
        """
        if pos:
            return self.lemmatizer.lemmatize(word.lower(), pos=pos)
        else:
            return self.lemmatizer.lemmatize(word.lower())
    
    def is_stop_word(self, word: str) -> bool:
        """
        Check if a word is a stop word.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is a stop word
        """
        return word.lower() in self.stop_words
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stop words removed
        """
        return [token for token in tokens if not self.is_stop_word(token)]
    
    def normalize_word(self, word: str) -> str:
        """
        Normalize a single word (lemmatize and convert to lowercase).
        
        Args:
            word: Word to normalize
            
        Returns:
            Normalized word
        """
        if word.isdecimal() or word.isdigit():
            return 'NUM'
        elif word in string.punctuation:
            return ''
        elif word in self.stop_words:
            return ''
        else:
            return self.lemmatizer.lemmatize(word.lower()) 