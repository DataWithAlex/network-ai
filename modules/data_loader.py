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

class BaseDataset:
    """Base class for dataset processing"""
    def __init__(self, test_size=0.2, random_state=42, batch_size=32):
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_names = None
        self.n_features = None
        self.n_classes = None
        
    def load_data(self):
        """Override this method in subclasses to load specific dataset"""
        raise NotImplementedError
        
    def get_dataloaders(self, train_dataset, test_dataset):
        """Create DataLoaders from datasets"""
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, test_loader
    
    def get_dataset_info(self):
        """Return information about the dataset"""
        return {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "description": self.get_description()
        }
    
    def get_description(self):
        """Return a description of the dataset"""
        return "Base dataset - implement in subclass"

class IrisDataset(BaseDataset):
    """Class for handling the Iris dataset"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_description(self):
        return """
        The Iris dataset consists of 150 samples from three species of Iris flowers.
        Each sample has four features: sepal length, sepal width, petal length, and petal width.
        This is a classic dataset for classification tasks.
        """
        
    def load_data(self):
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
        
        train_loader, test_loader = self.get_dataloaders(train_dataset, test_dataset)
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "original_X": X,
            "original_y": y
        }

class TitanicDataset(BaseDataset):
    """Class for handling the Titanic dataset"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_description(self):
        return """
        The Titanic dataset contains information about passengers aboard the RMS Titanic, 
        including whether they survived or not. Features include age, sex, class, fare, etc.
        This is a binary classification task to predict survival.
        """
        
    def load_data(self):
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
        
        train_loader, test_loader = self.get_dataloaders(train_dataset, test_dataset)
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "original_X": X,
            "original_y": y
        }

def get_dataset(dataset_name, **kwargs):
    """Factory function to get the appropriate dataset class"""
    datasets = {
        "Iris": IrisDataset,
        "Titanic": TitanicDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(datasets.keys())}")
        
    return datasets[dataset_name](**kwargs) 