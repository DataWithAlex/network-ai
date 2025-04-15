import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActivationHook:
    def __init__(self):
        self.activations = {}
        
    def hook_fn(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def clear(self):
        self.activations = {}

class BaseModel(nn.Module):
    """Base model class with common functionality"""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.activation_hook = ActivationHook()
        
    def get_intermediate_features(self, x):
        """Get activations at each layer for visualization"""
        self.activation_hook.clear()
        _ = self.forward(x)
        return self.activation_hook.activations
    
    def get_hyperparameters(self):
        """Return model hyperparameters for visualization"""
        raise NotImplementedError
    
    def get_description(self):
        """Return a description of the model"""
        raise NotImplementedError

class MLPClassifier(BaseModel):
    """Multi-layer Perceptron for classification"""
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU, dropout_rate=0.0):
        super(MLPClassifier, self).__init__()
        
        # Define network architecture
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation.__name__ if hasattr(activation, "__name__") else str(activation)
        self.dropout_rate = dropout_rate
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append((f'linear_{i}', nn.Linear(prev_size, hidden_size)))
            layers.append((f'activation_{i}', activation()))
            if dropout_rate > 0:
                layers.append((f'dropout_{i}', nn.Dropout(dropout_rate)))
            prev_size = hidden_size
            
        layers.append((f'linear_output', nn.Linear(prev_size, output_size)))
        
        self.model = nn.Sequential(
            *[layer for _, layer in layers]
        )
        
        # For storing the layer names and activation hooks
        self.layer_names = [name for name, _ in layers]
        self.register_hooks()
        
    def register_hooks(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(
                    self.activation_hook.hook_fn(self.layer_names[i])
                )
                
    def forward(self, x):
        return self.model(x)
    
    def get_layer_weights(self):
        """Extract weights from each layer for visualization"""
        weights = {}
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                weights[self.layer_names[i]] = {
                    'weight': layer.weight.detach().numpy(),
                    'bias': layer.bias.detach().numpy() if layer.bias is not None else None
                }
        return weights
    
    def get_hyperparameters(self):
        """Return model hyperparameters for visualization"""
        return {
            "Model Type": "Multi-layer Perceptron (MLP)",
            "Input Size": self.input_size,
            "Hidden Layers": self.hidden_sizes,
            "Output Size": self.output_size,
            "Activation Function": self.activation_name,
            "Dropout Rate": self.dropout_rate,
            "Total Parameters": sum(p.numel() for p in self.parameters()),
            "Trainable Parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_description(self):
        """Return a description of the model"""
        return f"""
        This is a Multi-layer Perceptron (MLP) classifier with {len(self.hidden_sizes)} hidden layers.
        
        Architecture:
        - Input layer: {self.input_size} neurons
        - Hidden layers: {', '.join(map(str, self.hidden_sizes))} neurons
        - Output layer: {self.output_size} neurons
        - Activation function: {self.activation_name}
        - Dropout rate: {self.dropout_rate}
        
        Total parameters: {sum(p.numel() for p in self.parameters())}
        """

def create_model(model_type, **kwargs):
    """Factory function to create a neural network model"""
    models = {
        "MLP": MLPClassifier
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Available models: {list(models.keys())}")
        
    return models[model_type](**kwargs) 