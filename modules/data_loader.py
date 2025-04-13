import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_names = None
        self.n_features = None
        self.n_classes = None
        
    def load_iris(self):
        """Load and preprocess the Iris dataset"""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        self.feature_names = iris.feature_names
        self.label_names = iris.target_names
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create dataset and dataloader
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "original_X": X,
            "original_y": y
        }
    
    def load_titanic(self):
        """Load and preprocess the Titanic dataset"""
        # URL for the Titanic dataset
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        
        # Preprocessing steps specific to Titanic
        # Drop unnecessary columns
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        
        # Handle missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        # Convert categorical features
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        
        # Extract features and labels
        y = df['Survived'].values
        X = df.drop('Survived', axis=1).values
        
        self.feature_names = df.drop('Survived', axis=1).columns.tolist()
        self.label_names = ['Not Survived', 'Survived']
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create dataset and dataloader
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "original_X": X,
            "original_y": y
        }
        
    def get_dataloaders(self, train_dataset, test_dataset, batch_size=32):
        """Create DataLoaders from datasets"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader 