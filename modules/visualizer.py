import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import math
import colorsys

# Import specialized visualization modules
from modules.interactive_visualizer import InteractiveNetworkVisualizer
from modules.activation_tracer import ActivationTracer

class NetworkVisualizer:
    """
    Main visualization coordinator class that delegates to specialized visualizers
    """
    def __init__(self, model, feature_names, label_names):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        
        # Initialize specialized visualizers
        self.interactive_viz = InteractiveNetworkVisualizer(model, feature_names, label_names)
        self.activation_tracer = ActivationTracer(model, feature_names, label_names)
        
        # Define colors for different layer types
        self.layer_colors = {
            'input': 'rgba(65, 105, 225, 0.7)',  # Royal blue
            'hidden': 'rgba(50, 205, 50, 0.7)',  # Lime green
            'output': 'rgba(220, 20, 60, 0.7)',  # Crimson
            'activation': 'rgba(147, 112, 219, 0.6)',  # Medium purple
            'dropout': 'rgba(169, 169, 169, 0.6)',  # Dark gray
            'batch_norm': 'rgba(255, 215, 0, 0.6)',  # Gold
            'fc': 'rgba(50, 205, 50, 0.7)',  # Same as hidden for fully connected layers
        }
        
        # Define shapes for different layer types (using valid Plotly shape types)
        self.layer_shapes = {
            'input': 'rect',
            'hidden': 'rect',
            'output': 'rect',
            'activation': 'circle',
            'dropout': 'rect',
            'batch_norm': 'rect',
            'fc': 'rect',  # Same as hidden for fully connected layers
        }
        
        # Weight color scales
        self.weight_pos_color = 'rgba(0, 128, 0, 0.7)'  # Green for positive weights
        self.weight_neg_color = 'rgba(255, 0, 0, 0.7)'  # Red for negative weights
        
    def plot_network_interactive(self):
        """Create an interactive network visualization using Plotly that matches the original style"""
        # Extract model structure and weights
        hyperparams = self.model.get_hyperparameters()
        
        # For MLPs
        if "MLP" in hyperparams["Model Type"]:
            input_size = hyperparams["Input Size"]
            hidden_sizes = hyperparams["Hidden Layers"]
            output_size = hyperparams["Output Size"]
            activation_fn = hyperparams["Activation Function"]
            
            # Create figure
            fig = go.Figure()
            
            # Layer definitions
            layers = []
            # Add input layer
            layers.append({
                'name': 'Input',
                'type': 'input',
                'size': input_size,
                'color': self.layer_colors['input']
            })
            
            # Add hidden layers with activations
            for i, size in enumerate(hidden_sizes):
                # Linear layer
                layers.append({
                    'name': f'FC-{i+1}',
                    'type': 'fc',
                    'size': size,
                    'color': self.layer_colors['fc']
                })
                
                # Activation layer
                layers.append({
                    'name': f'ReLU-{i+1}',
                    'type': 'activation',
                    'size': size,
                    'color': self.layer_colors['activation']
                })
                
            # Add output layer
            layers.append({
                'name': 'Output',
                'type': 'output',
                'size': output_size,
                'color': self.layer_colors['output']
            })
            
            # Visualization parameters
            layer_width = 120 
            layer_spacing = 200  # Horizontal spacing between layers
            node_spacing = 40    # Vertical spacing between nodes
            max_display_nodes = 8  # Maximum number of nodes to display in a layer
            
            # Add title
            fig.update_layout(
                title={
                    'text': f"Neural Network Architecture: Multi-layer Perceptron (MLP)",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add legend at the top
            legend_items = [
                {'name': 'Input', 'color': self.layer_colors['input']},
                {'name': 'FC-1', 'color': self.layer_colors['fc']},
                {'name': 'ReLU-1', 'color': self.layer_colors['activation']}, 
                {'name': 'FC-2', 'color': self.layer_colors['fc']},
                {'name': 'ReLU-2', 'color': self.layer_colors['activation']},
                {'name': 'Output', 'color': self.layer_colors['output']}
            ]
            
            for i, item in enumerate(legend_items):
                # Add a "square" marker for each legend item
                fig.add_trace(go.Scatter(
                    x=[i], 
                    y=[1],
                    mode='markers',
                    marker=dict(
                        color=item['color'],
                        size=15,
                        symbol='square',
                        line=dict(width=1, color='rgba(0,0,0,0.5)')
                    ),
                    name=item['name'],
                    showlegend=True
                ))
            
            # Draw each layer with its neurons
            for i, layer in enumerate(layers):
                x_pos = i * layer_spacing
                
                # Layer header (title and node count)
                fig.add_annotation(
                    x=x_pos,
                    y=max_display_nodes * node_spacing + 50,
                    text=f"{layer['name']}<br>({layer['size']} nodes)",
                    showarrow=False,
                    font=dict(size=12)
                )
                
                # Display nodes (limited to max_display_nodes)
                display_nodes = min(layer['size'], max_display_nodes)
                for j in range(display_nodes):
                    y_pos = j * node_spacing
                    
                    # Add node marker
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers',
                        marker=dict(
                            color=layer['color'],
                            size=15,
                            symbol='square',
                            line=dict(width=1, color='rgba(0,0,0,0.5)')
                        ),
                        hoverinfo='text',
                        hovertext=f"{layer['name']} Node {j+1}",
                        showlegend=False
                    ))
                
                # Add "..." to indicate more nodes if needed
                if layer['size'] > max_display_nodes:
                    fig.add_annotation(
                        x=x_pos,
                        y=(display_nodes + 0.5) * node_spacing,
                        text="...",
                        showarrow=False,
                        font=dict(size=16)
                    )
            
            # Set axis properties
            fig.update_layout(
                width=1000,
                height=600,
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                margin=dict(t=100, b=20, l=20, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            return fig
        else:
            # Placeholder for other model types
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Interactive visualization not implemented for {hyperparams['Model Type']}",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                width=800,
                height=600
            )
            return fig
    
    def create_interactive_activation_viz(self, sample_input, sample_label=None):
        """Delegate to activation tracer"""
        return self.activation_tracer.create_interactive_activation_viz(sample_input, sample_label)
    
    def plot_detailed_network(self, sample_input=None, sample_label=None):
        """Create detailed network visualization with sample propagation"""
        return self.activation_tracer.visualize_network_flow(sample_input, sample_label)
    
    def plot_activation_heatmap(self, sample_input):
        """Create heatmap of activations for a sample input"""
        # Get activations
        activation_info = self.activation_tracer.trace_sample(sample_input)
        activations = activation_info["activations"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        layer_names = []
        activation_data = []
        
        for layer_name, act in activations.items():
            if 'linear' in layer_name:  # Only show linear layer outputs
                layer_names.append(layer_name)
                # Get first sample in batch
                activation_data.append(act[0].detach().numpy())
        
        # Create heatmap
        if activation_data:
            # Combine all activations into one array
            all_acts = np.concatenate(activation_data)
            
            # Set a common color scale
            vmin = np.min(all_acts)
            vmax = np.max(all_acts)
            
            # Plot each layer's activations
            for i, (name, acts) in enumerate(zip(layer_names, activation_data)):
                ax.text(-0.5, i, name, ha='right', va='center')
                img = ax.imshow(acts.reshape(1, -1), aspect='auto', 
                              cmap='viridis', vmin=vmin, vmax=vmax,
                              extent=[0, acts.shape[0], i-0.4, i+0.4])
                
                # Add text annotations for each activation value
                if acts.shape[0] <= 20:  # Only show values for small layers
                    for j, val in enumerate(acts):
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                              fontsize=8, color='white' if val < (vmax+vmin)/2 else 'black')
            
            # Add colorbar
            plt.colorbar(img, ax=ax, orientation='vertical', pad=0.01)
            
            # Set labels and title
            ax.set_title('Activation Values Across Layers')
            ax.set_xlabel('Neuron Index')
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels([])  # Hide y-tick labels since we add text directly
            
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "No activation data available", 
                  ha='center', va='center', transform=ax.transAxes)
            
        return fig
    
    def visualize_feature_importance(self, X, y, method='weights'):
        """Visualize feature importance based on input layer weights"""
        weights = self.model.get_layer_weights()
        
        # Get the first layer weights
        first_layer_name = list(weights.keys())[0]
        first_layer_weights = weights[first_layer_name]['weight']
        
        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.sum(np.abs(first_layer_weights), axis=0)
        
        # Normalize to get relative importance
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create a DataFrame for visualization
        feature_names = self.feature_names if len(self.feature_names) == len(feature_importance) else [f"Feature {i+1}" for i in range(len(feature_importance))]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        
        # Add labels and title
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance Based on First Layer Weights')
        
        # Add a grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def visualize_embeddings(self, X, y, latent_dim=2, method='tsne'):
        """Visualize data embeddings using dimensionality reduction"""
        if not torch.is_tensor(X):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
            
        # Get intermediate layer activations
        layer_outputs = {}
        for i in range(X_tensor.shape[0]):
            sample = X_tensor[i:i+1]
            activations = self.model.get_intermediate_features(sample)
            
            for name, activation in activations.items():
                if name not in layer_outputs:
                    layer_outputs[name] = []
                layer_outputs[name].append(activation[0].detach().numpy())
        
        # Convert lists to numpy arrays
        for name in layer_outputs:
            layer_outputs[name] = np.vstack(layer_outputs[name])
        
        # Create multi-panel visualization
        n_layers = len(layer_outputs)
        fig, axes = plt.subplots(1, n_layers+1, figsize=(4*(n_layers+1), 4), squeeze=False)
        axes = axes.flatten()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            tsne = TSNE(n_components=latent_dim, random_state=42)
            
            # Plot original data
            X_tsne = tsne.fit_transform(X)
            
            scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                                     s=50, alpha=0.8, edgecolors='w')
            
            # Add a legend to the first plot
            legend1 = axes[0].legend(*scatter.legend_elements(),
                                   loc="upper right", title="Classes")
            axes[0].add_artist(legend1)
            
            axes[0].set_xlabel('t-SNE 1')
            axes[0].set_ylabel('t-SNE 2')
            axes[0].set_title('Input Data')
            
            # Plot layer outputs
            for i, (name, outputs) in enumerate(layer_outputs.items()):
                layer_tsne = tsne.fit_transform(outputs)
                
                scatter = axes[i+1].scatter(layer_tsne[:, 0], layer_tsne[:, 1], c=y, 
                                         cmap='viridis', s=50, alpha=0.8, edgecolors='w')
                
                axes[i+1].set_xlabel('t-SNE 1')
                axes[i+1].set_ylabel('t-SNE 2')
                axes[i+1].set_title(f't-SNE: {name}')
        
        elif method.lower() == 'pca':
            # Similar implementation for PCA
            pca = PCA(n_components=latent_dim)
            
            # Plot original data
            X_pca = pca.fit_transform(X)
            
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                                     s=50, alpha=0.8, edgecolors='w')
            
            # Add explained variance as title
            var = pca.explained_variance_ratio_
            axes[0].set_title(f'PCA: Input Data\nExplained var: {var[0]:.2f}, {var[1]:.2f}')
            
            # Add a legend to the first plot
            legend1 = axes[0].legend(*scatter.legend_elements(),
                                   loc="upper right", title="Classes")
            axes[0].add_artist(legend1)
            
            axes[0].set_xlabel('PC 1')
            axes[0].set_ylabel('PC 2')
            
            # Plot layer outputs
            for i, (name, outputs) in enumerate(layer_outputs.items()):
                pca = PCA(n_components=latent_dim)
                layer_pca = pca.fit_transform(outputs)
                var = pca.explained_variance_ratio_
                
                scatter = axes[i+1].scatter(layer_pca[:, 0], layer_pca[:, 1], c=y, 
                                         cmap='viridis', s=50, alpha=0.8, edgecolors='w')
                
                axes[i+1].set_xlabel('PC 1')
                axes[i+1].set_ylabel('PC 2')
                axes[i+1].set_title(f'PCA: {name}\nExplained var: {var[0]:.2f}, {var[1]:.2f}')
        
        plt.tight_layout()
        return fig
    
    def visualize_decision_boundary(self, X, y, resolution=0.01):
        """Visualize decision boundary for 2D data"""
        # This only works for 2 features
        if X.shape[1] != 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Decision boundary visualization requires 2D data", 
                  ha='center', va='center', transform=ax.transAxes)
            return fig
            
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define mesh grid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))
        
        # Create tensor of all points in the mesh
        Z = np.c_[xx.ravel(), yy.ravel()]
        Z_tensor = torch.tensor(Z, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(Z_tensor)
            predicted = torch.argmax(logits, dim=1).numpy()
        
        # Plot decision boundary
        predicted = predicted.reshape(xx.shape)
        ax.contourf(xx, yy, predicted, alpha=0.3, cmap='viridis')
        
        # Plot training points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                            edgecolor='black', s=50)
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(),
                           loc="upper right", title="Classes")
        ax.add_artist(legend1)
        
        # Add labels
        if len(self.feature_names) >= 2:
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
        else:
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
        
        ax.set_title('Decision Boundary')
        return fig

    def visualize_pca_projection(self, X, y, layer_outputs=None):
        """Visualize PCA projection of the data and layer outputs"""
        if layer_outputs is None:
            # Just visualize the input data
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                                s=50, alpha=0.8, edgecolors='w')
            
            # Add a legend
            legend1 = ax.legend(*scatter.legend_elements(),
                              loc="upper right", title="Classes")
            ax.add_artist(legend1)
            
            # Add labels and title
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title('PCA Projection of Input Data')
            
            return fig
        else:
            # Visualize projections at each layer
            fig, axes = plt.subplots(1, len(layer_outputs) + 1, figsize=(5 * (len(layer_outputs) + 1), 5))
            
            # Plot input data projection
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                                     s=50, alpha=0.8, edgecolors='w')
            
            # Add a legend to the first plot
            legend1 = axes[0].legend(*scatter.legend_elements(),
                                   loc="upper right", title="Classes")
            axes[0].add_artist(legend1)
            
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].set_title('Input Data')
            
            # Plot layer outputs
            for i, (name, outputs) in enumerate(layer_outputs.items()):
                # For high-dimensional layers, use PCA
                pca = PCA(n_components=2)
                layer_pca = pca.fit_transform(outputs.detach().numpy())
                
                scatter = axes[i+1].scatter(layer_pca[:, 0], layer_pca[:, 1], c=y, 
                                         cmap='viridis', s=50, alpha=0.8, edgecolors='w')
                
                axes[i+1].set_xlabel('PC1')
                axes[i+1].set_ylabel('PC2')
                axes[i+1].set_title(f'PCA: {name}')
            
            plt.tight_layout()
            return fig

    def visualize_tsne_projection(self, X, y, layer_outputs=None, perplexity=30):
        """Visualize t-SNE projection of the data and layer outputs"""
        if layer_outputs is None:
            # Just visualize the input data
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                                s=50, alpha=0.8, edgecolors='w')
            
            # Add a legend
            legend1 = ax.legend(*scatter.legend_elements(),
                                  loc="upper right", title="Classes")
            ax.add_artist(legend1)
            
            # Add labels and title
            ax.set_xlabel('t-SNE Feature 1')
            ax.set_ylabel('t-SNE Feature 2')
            ax.set_title('t-SNE Projection of Input Data')
            
            return fig
        else:
            # Visualize projections at each layer
            fig, axes = plt.subplots(1, len(layer_outputs) + 1, figsize=(5 * (len(layer_outputs) + 1), 5))
            
            # Plot input data projection
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                                     s=50, alpha=0.8, edgecolors='w')
            
            # Add a legend to the first plot
            legend1 = axes[0].legend(*scatter.legend_elements(),
                                   loc="upper right", title="Classes")
            axes[0].add_artist(legend1)
            
            axes[0].set_xlabel('t-SNE 1')
            axes[0].set_ylabel('t-SNE 2')
            axes[0].set_title('Input Data')
            
            # Plot layer outputs
            for i, (name, outputs) in enumerate(layer_outputs.items()):
                layer_tsne = tsne.fit_transform(outputs.detach().numpy())
                
                scatter = axes[i+1].scatter(layer_tsne[:, 0], layer_tsne[:, 1], c=y, 
                                         cmap='viridis', s=50, alpha=0.8, edgecolors='w')
                
                axes[i+1].set_xlabel('t-SNE 1')
                axes[i+1].set_ylabel('t-SNE 2')
                axes[i+1].set_title(f't-SNE: {name}')
            
            plt.tight_layout()
            return fig

    def visualize_activations(self, input_data):
        """Visualize activations at each layer for a specific input example"""
        if isinstance(input_data, torch.Tensor):
            x = input_data.clone()
        else:
            x = torch.tensor(input_data, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Get activations
        activations = self.model.get_intermediate_features(x)
        
        # Create figure
        num_layers = len(activations)
        fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
        
        # If there's only one layer, wrap axes in a list
        if num_layers == 1:
            axes = [axes]
        
        # Plot activations for each layer
        for i, (layer_name, activation) in enumerate(activations.items()):
            act_data = activation.squeeze().detach().numpy()
            
            if act_data.ndim == 1:  # For 1D activations (e.g., fully connected layers)
                axes[i].bar(range(len(act_data)), act_data)
                axes[i].set_title(f"{layer_name}")
                axes[i].set_xlabel("Neuron")
                axes[i].set_ylabel("Activation")
            
            elif act_data.ndim == 2:  # For 2D activations (e.g., convolutional layers)
                im = axes[i].imshow(act_data, cmap='viridis')
                axes[i].set_title(f"{layer_name}")
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        return fig