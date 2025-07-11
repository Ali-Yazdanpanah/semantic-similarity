# Modular Semantic Textual Similarity (STS) System

A comprehensive, modular system for computing semantic similarity between sentences using multiple approaches including traditional machine learning, neural networks, and BERT embeddings.

## 🚀 Features

- **Multiple Approaches**: Traditional ML, Neural Networks, and BERT embeddings
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Evaluation**: Multiple metrics and detailed analysis
- **Interactive Mode**: Test sentence pairs in real-time
- **Easy Configuration**: Command-line arguments for different modes

## 📁 Project Structure

```
NLP/
├── data/
│   ├── __init__.py
│   └── dataset_loader.py          # Dataset loading and preprocessing
├── features/
│   ├── __init__.py
│   ├── text_preprocessor.py       # Text preprocessing and normalization
│   └── similarity_features.py     # Traditional similarity features
├── models/
│   ├── __init__.py
│   ├── neural_network.py          # PyTorch neural network model
│   └── bert_embeddings.py         # BERT embeddings and features
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 # Evaluation metrics and analysis
├── sts_similarity_modular.py      # Main modular system
├── example_usage.py               # Example usage demonstrations
├── download_sts_dataset.py        # Dataset downloader
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already installed):
   ```python3
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('averaged_perceptron_tagger_eng')
   nltk.download('wordnet')
   nltk.download('wordnet_ic')
   nltk.download('brown')
   ```

4. **Download the STS dataset**:
   ```bash
   python3 download_sts_dataset.py
   ```

## 🎯 Usage

### Basic Usage (Traditional Features + Neural Network)

```bash
python3 sts_similarity_modular.py
```

This runs the complete experiment with:
- Traditional similarity features (WordNet, TF-IDF, POS-based features)
- Neural network model
- Multiple traditional ML models (SVR, Random Forest, Linear Regression)

### Advanced Usage (with BERT)

```bash
python3 sts_similarity_modular.py --use-bert
```

This enables BERT embeddings for potentially better performance (but slower execution).

### Interactive Mode

Test sentence pairs interactively:

```bash
python3 sts_similarity_modular.py --interactive
```

### Disable Neural Network

```bash
python3 sts_similarity_modular.py --no-neural
```

## 📊 Features Overview

### Text Preprocessing (`features/text_preprocessor.py`)
- **Tokenization**: TweetTokenizer for robust tokenization
- **Lemmatization**: WordNet-based lemmatization
- **Stop Word Removal**: Comprehensive stop word list
- **Text Normalization**: Number replacement, punctuation handling

### Traditional Features (`features/similarity_features.py`)
- **Simple Baseline**: Word overlap with TF-IDF weighting
- **WordNet Similarity**: Path similarity and information content
- **POS-based Features**: Part-of-speech distribution differences
- **Unigram Overlap**: Word-level overlap measures

### Neural Network (`models/neural_network.py`)
- **PyTorch Implementation**: Multi-layer perceptron with dropout
- **Early Stopping**: Prevents overfitting
- **GPU Support**: Automatic CUDA detection
- **Batch Processing**: Efficient training

### BERT Embeddings (`models/bert_embeddings.py`)
- **Contextual Embeddings**: BERT-based sentence representations
- **Multiple Similarity Measures**: Cosine, Euclidean, Manhattan distances
- **Batch Processing**: Efficient feature extraction
- **Error Handling**: Graceful fallback for failed extractions

### Evaluation (`evaluation/metrics.py`)
- **Multiple Metrics**: Pearson, Spearman, MSE, RMSE, MAE
- **Model Comparison**: Side-by-side performance comparison
- **Prediction Analysis**: Best and worst prediction examples
- **Score Distribution**: Statistical analysis of predictions

## 🔧 Configuration

### Model Parameters

You can modify model parameters in the respective files:

- **Neural Network**: `models/neural_network.py` - Adjust hidden layers, learning rate, etc.
- **BERT**: `models/bert_embeddings.py` - Change model name, max length, etc.
- **Traditional Features**: `features/similarity_features.py` - Modify feature extraction

### Dataset Configuration

- **Data Directory**: Modify `data_dir` in `STSDatasetLoader`
- **File Names**: Change expected file names in `load_all_datasets()`

## 📈 Expected Results

With the STS-B dataset, you can expect:

- **Traditional Features**: Pearson correlation ~0.65-0.75
- **Neural Network**: Pearson correlation ~0.70-0.80
- **BERT Embeddings**: Pearson correlation ~0.80-0.85

## 🎓 Educational Value

This project demonstrates:

1. **Modular Software Design**: Clean separation of data, features, models, and evaluation
2. **Multiple ML Approaches**: Traditional ML vs. deep learning
3. **NLP Techniques**: WordNet, POS tagging, embeddings
4. **Evaluation Best Practices**: Multiple metrics, error analysis
5. **PyTorch Integration**: Modern deep learning workflows

## 🔍 Example Output

```
MODULAR STS SYSTEM
==================================================
Loading datasets...

==================================================
DATASET INFORMATION
==================================================

TRAIN SET:
  Number of pairs: 5749
  Score range: 0.00 - 5.00
  Average score: 2.50
  Score std dev: 1.13
  Average sentence 1 length: 10.2 words
  Average sentence 2 length: 10.1 words

DEV SET:
  Number of pairs: 1500
  Score range: 0.00 - 5.00
  Average score: 2.50
  Score std dev: 1.13
  Average sentence 1 length: 10.2 words
  Average sentence 2 length: 10.1 words

Vocabulary size: 12345 unique words

==================================================
MODEL EVALUATION
================================================================================

Evaluating on train set...

==================================================
EVALUATION RESULTS - TRAIN - SVR
==================================================
Pearson Correlation:  0.7234
Spearman Correlation: 0.7156
Mean Squared Error:   0.8234
Root MSE:             0.9074
Mean Absolute Error:  0.7234
==================================================

==================================================
MODEL COMPARISON
================================================================================
Model                Pearson    Spearman   RMSE       MAE       
--------------------------------------------------------------------------------
SVR                  0.7234     0.7156     0.9074     0.7234    
RandomForest         0.6987     0.6923     0.9456     0.7567    
LinearRegression     0.6543     0.6489     1.0234     0.8234    
NeuralNetwork        0.7456     0.7389     0.8765     0.6987    
================================================================================
```

## 🤝 Contributing

Feel free to extend the system by:

1. **Adding New Features**: Implement additional similarity measures
2. **New Models**: Add different neural architectures
3. **Additional Datasets**: Support for other STS datasets
4. **Visualization**: Add plotting and analysis tools

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- STS-B dataset from Facebook Research
- NLTK for NLP tools
- PyTorch for deep learning
- Hugging Face Transformers for BERT
- Scikit-learn for traditional ML 