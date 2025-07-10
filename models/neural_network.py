"""
Neural Network Model for STS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional


class STSDataset(Dataset):
    """PyTorch Dataset for STS data."""
    
    def __init__(self, features: np.ndarray, scores: Optional[List[float]] = None):
        self.features = torch.FloatTensor(features)
        if scores is not None:
            self.scores = torch.FloatTensor(scores)
        else:
            self.scores = torch.zeros(len(features))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'score': self.scores[idx]
        }


class STSNeuralNetwork(nn.Module):
    """Neural Network for STS prediction."""
    
    def __init__(self, input_size: int = 14, hidden_sizes: List[int] = [128, 64, 32], dropout_rate: float = 0.3):
        super(STSNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()


class NeuralSTSModel:
    """Neural Network-based STS Model."""
    
    def __init__(self, input_size: int = 14, hidden_sizes: List[int] = [128, 64, 32], 
                 learning_rate: float = 0.001, batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STSNeuralNetwork(input_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        
    def train(self, train_features: np.ndarray, train_scores: List[float], 
              dev_features: np.ndarray, dev_scores: List[float], 
              epochs: int = 50, patience: int = 10):
        """Train the neural network."""
        
        # Create datasets
        train_dataset = STSDataset(train_features, train_scores)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        dev_dataset = STSDataset(dev_features, dev_scores)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_dev_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(dev_dataset)}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                features = batch['features'].to(self.device)
                scores = batch['score'].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, scores)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            dev_loss = 0.0
            dev_predictions = []
            dev_true_scores = []
            
            with torch.no_grad():
                for batch in dev_loader:
                    features = batch['features'].to(self.device)
                    scores = batch['score'].to(self.device)
                    
                    predictions = self.model(features)
                    loss = self.criterion(predictions, scores)
                    dev_loss += loss.item()
                    
                    dev_predictions.extend(predictions.cpu().numpy())
                    dev_true_scores.extend(scores.cpu().numpy())
            
            # Calculate Pearson correlation
            from scipy.stats import pearsonr
            correlation = pearsonr(dev_true_scores, dev_predictions)[0]
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Dev Loss: {dev_loss/len(dev_loader):.4f}, Dev Correlation: {correlation:.4f}")
            
            # Early stopping
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        dataset = STSDataset(features)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                features_batch = batch['features'].to(self.device)
                preds = self.model(features_batch)
                predictions.extend(preds.cpu().numpy())
        
        # Clip predictions to [0, 5] range
        predictions = np.array(predictions)
        predictions = np.clip(predictions, 0, 5)
        
        return predictions 