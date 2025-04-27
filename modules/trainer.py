import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Trainer:
    def __init__(self, model, learning_rate=0.001, criterion=None):
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy
    
    def train(self, train_loader, val_loader, epochs=50, verbose=True):
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Save history
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
                
        return self.training_history
    
    def get_predictions(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.numpy())
                
        return np.array(all_preds), np.array(all_targets), np.array(all_probs)
    
    def get_performance_metrics(self, loader, label_names=None):
        predictions, targets, probabilities = self.get_predictions(loader)
        
        accuracy = accuracy_score(targets, predictions)
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Get classification report as text and as dictionary
        report = classification_report(targets, predictions, target_names=label_names)
        report_dict = classification_report(targets, predictions, target_names=label_names, output_dict=True)
        
        # Return all necessary data for visualizations including ROC curve data
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'classification_report_dict': report_dict,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            # Add these for ROC curve
            'y_true': targets,
            'y_score': probabilities
        } 