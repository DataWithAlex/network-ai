import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationHook:
    def __init__(self):
        self.activations = {}
        
    def hook_fn(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def clear(self):
        self.activations = {}

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU):
        super(SimpleNN, self).__init__()
        
        # Define network architecture
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append((f'linear_{i}', nn.Linear(prev_size, hidden_size)))
            layers.append((f'activation_{i}', activation()))
            prev_size = hidden_size
            
        layers.append((f'linear_output', nn.Linear(prev_size, output_size)))
        
        self.model = nn.Sequential(
            *[layer for _, layer in layers]
        )
        
        # For storing the layer names and activation hooks
        self.layer_names = [name for name, _ in layers]
        self.activation_hook = ActivationHook()
        self.register_hooks()
        
    def register_hooks(self):
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(
                    self.activation_hook.hook_fn(self.layer_names[i])
                )
                
    def get_intermediate_features(self, x):
        """Get activations at each layer for visualization"""
        self.activation_hook.clear()
        _ = self.forward(x)
        return self.activation_hook.activations
    
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

def create_model(input_size, hidden_sizes, output_size):
    """Factory function to create a neural network model"""
    model = SimpleNN(input_size, hidden_sizes, output_size)
    return model 