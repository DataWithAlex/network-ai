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
        """
        Create an interactive visualization of the neural network architecture
        with clear representation of activation functions.
        
        Returns:
            Plotly figure object
        """
        # Get model structure
        if hasattr(self.model, 'get_network_structure'):
            network_structure = self.model.get_network_structure()
        else:
            # Fallback to manual structure extraction
            network_structure = self._extract_network_structure()
        
        # Extract layer weights for connection visualization
        weights = self.model.get_layer_weights()
        
        # Create nodes and edges for the graph
        nodes = []
        edges = []
        
        # Define layer colors with better contrast
        layer_colors = {
            'input': 'rgba(65, 105, 225, 0.8)',  # Royal blue
            'linear': 'rgba(60, 179, 113, 0.8)',  # Medium sea green
            'activation': 'rgba(147, 112, 219, 0.8)',  # Medium purple
            'output': 'rgba(220, 20, 60, 0.8)'   # Crimson
        }
        
        # Calculate total number of layers for horizontal spacing
        total_layers = len(network_structure)
        
        # Create nodes for each layer
        node_id = 0
        layer_nodes = {}  # Store nodes by layer for edge creation
        
        # First pass: create nodes
        for i, layer in enumerate(network_structure):
            layer_type = layer['type']
            layer_name = layer['name']
            num_nodes = layer['nodes']
            
            # Calculate x position (evenly spaced)
            x_pos = i / (total_layers - 1) if total_layers > 1 else 0.5
            
            # Create a list to store node IDs for this layer
            layer_nodes[layer_name] = []
            
            # Create nodes for this layer
            for j in range(num_nodes):
                # Calculate y position to center nodes vertically
                y_pos = j - (num_nodes - 1) / 2
                
                # Determine node label based on layer type
                if layer_type == 'input' and j < len(self.feature_names):
                    label = self.feature_names[j]
                elif layer_type == 'output' and j < len(self.label_names):
                    label = self.label_names[j]
                elif layer_type == 'activation':
                    # Show activation function type (e.g., ReLU)
                    label = f"{layer_name}"
                else:
                    label = f"{layer_name}<br>Node {j+1}" if num_nodes > 1 else layer_name
                
                # Create node with appropriate styling
                node_info = {
                    'id': node_id,
                    'label': label,
                    'x': x_pos,
                    'y': y_pos,
                    'size': 30,
                    'color': layer_colors.get(layer_type, 'rgba(128, 128, 128, 0.8)'),
                    'layer_type': layer_type,
                    'layer_name': layer_name,
                    'node_index': j
                }
                
                nodes.append(node_info)
                layer_nodes[layer_name].append(node_id)
                node_id += 1
        
        # Second pass: create edges between layers
        for i in range(len(network_structure) - 1):
            current_layer = network_structure[i]
            next_layer = network_structure[i + 1]
            
            current_layer_name = current_layer['name']
            next_layer_name = next_layer['name']
            
            # Get weight matrix if available (for linear layers)
            weight_key = current_layer_name if current_layer_name in weights else None
            
            # If this is a linear layer connecting to an activation layer, use special styling
            is_activation_connection = next_layer['type'] == 'activation'
            
            # Connect each node in current layer to each node in next layer
            for src_idx, src_node_id in enumerate(layer_nodes[current_layer_name]):
                for dst_idx, dst_node_id in enumerate(layer_nodes[next_layer_name]):
                    # Determine edge weight and color
                    if weight_key and not is_activation_connection:
                        # Use actual weights for connections between computation layers
                        weight_matrix = weights[weight_key]['weight']
                        if src_idx < weight_matrix.shape[1] and dst_idx < weight_matrix.shape[0]:
                            weight = weight_matrix[dst_idx, src_idx]
                            # Normalize weight for visualization
                            weight_abs = abs(weight)
                            # Color based on sign (green for positive, red for negative)
                            edge_color = f'rgba(0, 128, 0, {min(0.8, 0.2 + weight_abs)})' if weight > 0 else f'rgba(255, 0, 0, {min(0.8, 0.2 + weight_abs)})'
                            width = 1 + 3 * weight_abs
                        else:
                            # Default for out-of-range indices
                            weight = 0
                            edge_color = 'rgba(200, 200, 200, 0.3)'
                            width = 1
                    elif is_activation_connection:
                        # For connections to activation layers, use a distinctive style
                        # to show these are not weighted connections
                        weight = 1.0
                        edge_color = 'rgba(100, 100, 255, 0.7)'  # Blue for activation connections
                        width = 2
                    else:
                        # Default for layers without weights
                        weight = 0.5
                        edge_color = 'rgba(200, 200, 200, 0.5)'
                        width = 1
                    
                    # Create edge
                    edge_info = {
                        'source': src_node_id,
                        'target': dst_node_id,
                        'weight': weight,
                        'color': edge_color,
                        'width': width,
                        'is_activation': is_activation_connection
                    }
                    
                    edges.append(edge_info)
        
        # Create edge traces
        edge_trace = []
        
        # Group edges by type for better visualization
        edge_groups = {
            'weights': [e for e in edges if not e['is_activation']],
            'activations': [e for e in edges if e['is_activation']]
        }
        
        # Create traces for each edge group
        for group_name, group_edges in edge_groups.items():
            for edge in group_edges:
                source_node = nodes[edge['source']]
                target_node = nodes[edge['target']]
                
                # Create line trace
                trace = go.Scatter(
                    x=[source_node['x'], target_node['x']],
                    y=[source_node['y'], target_node['y']],
                    mode='lines',
                    line=dict(
                        width=edge['width'],
                        color=edge['color'],
                        dash='dot' if edge['is_activation'] else 'solid'  # Dotted lines for activation connections
                    ),
                    hoverinfo='text',
                    text=f"{'Activation' if edge['is_activation'] else 'Weight'}: {edge['weight']:.3f}",
                    showlegend=False
                )
                
                edge_trace.append(trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[node['x'] for node in nodes],
            y=[node['y'] for node in nodes],
            mode='markers+text',
            text=[node['label'] for node in nodes],
            textposition='middle center',
            hoverinfo='text',
            marker=dict(
                size=[node['size'] for node in nodes],
                color=[node['color'] for node in nodes],
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title='Neural Network Architecture: Multi-layer Perceptron (MLP)',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                annotations=[
                    dict(
                        x=0,
                        y=1.05,
                        xref='paper',
                        yref='paper',
                        text='Input',
                        showarrow=False,
                        font=dict(size=14)
                    ),
                    dict(
                        x=0.5,
                        y=1.05,
                        xref='paper',
                        yref='paper',
                        text='Hidden Layers',
                        showarrow=False,
                        font=dict(size=14)
                    ),
                    dict(
                        x=1,
                        y=1.05,
                        xref='paper',
                        yref='paper',
                        text='Output',
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
        )
        
        # Add a legend explaining the visualization
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref='paper',
            yref='paper',
            text='<b>Legend:</b> Blue nodes = Input, Green nodes = Linear layers, Purple nodes = Activation functions, Red nodes = Output<br>' +
                 'Green edges = Positive weights, Red edges = Negative weights, Blue dotted lines = Activation connections',
            showarrow=False,
            font=dict(size=12),
            align='center',
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            opacity=0.8
        )
        
        return fig

    def _extract_network_structure(self):
        """
        Fallback method to extract network structure if get_network_structure is not available.
        """
        # Basic structure extraction
        structure = []
        
        # Add input layer
        if hasattr(self.model, 'input_size'):
            structure.append({
                'name': 'Input',
                'type': 'input',
                'size': self.model.input_size,
                'nodes': self.model.input_size
            })
        
        # Add hidden layers
        if hasattr(self.model, 'hidden_sizes'):
            for i, size in enumerate(self.model.hidden_sizes):
                # Add linear layer
                structure.append({
                    'name': f'FC-{i+1}',
                    'type': 'linear',
                    'size': size,
                    'nodes': size
                })
                
                # Add activation layer
                structure.append({
                    'name': f'ReLU-{i+1}',
                    'type': 'activation',
                    'size': size,
                    'nodes': size
                })
        
        # Add output layer
        if hasattr(self.model, 'output_size'):
            structure.append({
                'name': 'Output',
                'type': 'output',
                'size': self.model.output_size,
                'nodes': self.model.output_size
            })
        
        return structure
    
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
    
    def visualize_feature_importance(self, feature_names=None):
        """
        Visualize feature importance based on the weights of the first layer.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            Plotly figure object
        """
        # Get model weights
        weights = self.model.get_layer_weights()
        
        # Get first layer weights
        first_layer_name = list(weights.keys())[0]
        first_layer_weights = weights[first_layer_name]['weight']
        
        # Calculate feature importance as the sum of absolute weights
        # Handle both PyTorch tensors and numpy arrays
        if torch.is_tensor(first_layer_weights):
            feature_importance = torch.sum(torch.abs(first_layer_weights), dim=0).detach().cpu().numpy()
        else:
            # If it's already a numpy array
            feature_importance = np.sum(np.abs(first_layer_weights), axis=0)
        
        # Normalize to sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Use provided feature names or default ones
        if feature_names is None:
            feature_names = self.feature_names if hasattr(self, 'feature_names') else [f"Feature {i}" for i in range(len(feature_importance))]
        
        # Create a DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Create the bar chart
        fig = px.bar(
            importance_df, 
            y='Feature', 
            x='Importance',
            orientation='h',
            title='Feature Importance Based on First Layer Weights',
            labels={'Importance': 'Relative Importance', 'Feature': ''},
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            title_font=dict(size=16)
        )
        
        return fig

    def perform_sensitivity_analysis(self, X, feature_names=None, n_samples=10):
        """
        Perform sensitivity analysis by varying each feature and measuring the impact on predictions.
        
        Args:
            X: Input data (numpy array or tensor)
            feature_names: List of feature names
            n_samples: Number of variations to test per feature
            
        Returns:
            Plotly figure object
        """
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.numpy()
        
        # Use provided feature names or default ones
        if feature_names is None:
            feature_names = self.feature_names if hasattr(self, 'feature_names') else [f"Feature {i}" for i in range(X.shape[1])]
        
        # Get baseline predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            baseline_outputs = self.model(X_tensor)
            baseline_probs = torch.softmax(baseline_outputs, dim=1).mean(dim=0).numpy()
        
        # Store sensitivity results
        sensitivity_results = []
        
        # For each feature
        for feature_idx in range(X.shape[1]):
            feature_name = feature_names[feature_idx]
            
            # Get feature min and max
            feature_min = np.min(X[:, feature_idx])
            feature_max = np.max(X[:, feature_idx])
            
            # Create variations
            variations = np.linspace(feature_min, feature_max, n_samples)
            
            # Store variation results
            variation_impacts = []
            
            # For each variation
            for variation in variations:
                # Create a copy of the data
                X_modified = X.copy()
                
                # Modify the feature
                X_modified[:, feature_idx] = variation
                
                # Get predictions
                with torch.no_grad():
                    X_modified_tensor = torch.tensor(X_modified, dtype=torch.float32)
                    outputs = self.model(X_modified_tensor)
                    probs = torch.softmax(outputs, dim=1).mean(dim=0).numpy()
                
                # Calculate impact as the mean absolute difference in probabilities
                impact = np.mean(np.abs(probs - baseline_probs))
                
                # Store result
                variation_impacts.append({
                    'Feature': feature_name,
                    'Value': variation,
                    'Impact': impact
                })
            
            # Add to results
            sensitivity_results.extend(variation_impacts)
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Create the line chart
        fig = px.line(
            sensitivity_df,
            x='Value',
            y='Impact',
            color='Feature',
            title='Feature Sensitivity Analysis',
            labels={
                'Value': 'Feature Value',
                'Impact': 'Impact on Prediction (Mean Absolute Difference)',
                'Feature': 'Feature'
            },
            line_shape='linear'
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
            title_font=dict(size=16)
        )
        
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

    def visualize_pca_projection(self, X, y, layer_outputs):
        """
        Visualize PCA projections of the data at different layers of the network.
        
        Args:
            X: Input data
            y: Labels
            layer_outputs: Dictionary of layer outputs
        
        Returns:
            Plotly figure with PCA projections
        """
        # Create subplots
        n_plots = len(layer_outputs) + 1  # Input data + each layer
        fig = make_subplots(rows=1, cols=n_plots, 
                            subplot_titles=["Input Data"] + list(layer_outputs.keys()))
        
        # Get unique classes and their names
        unique_classes = np.unique(y)
        
        # Ensure we have proper class names
        if hasattr(self, 'label_names') and len(self.label_names) >= len(unique_classes):
            class_names = [self.label_names[int(cls)] for cls in unique_classes]
        else:
            class_names = [f"Class {i}" for i in unique_classes]
        
        # Create a color map for the classes
        colors = px.colors.qualitative.Set1[:len(unique_classes)]
        
        # PCA for input data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Plot input data
        for i, cls in enumerate(unique_classes):
            mask = (y == cls)
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8),
                    name=class_names[i],  # Use proper class name
                    legendgroup=f"class_{cls}",  # Use consistent legend group
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # PCA for each layer
        for i, (layer_name, layer_output) in enumerate(layer_outputs.items()):
            # Convert to numpy if tensor
            if torch.is_tensor(layer_output):
                layer_output = layer_output.detach().cpu().numpy()
            
            # Reshape if needed
            if len(layer_output.shape) > 2:
                layer_output = layer_output.reshape(layer_output.shape[0], -1)
            
            # Apply PCA
            pca = PCA(n_components=2)
            layer_pca = pca.fit_transform(layer_output)
            
            # Plot each class with proper labels
            for j, cls in enumerate(unique_classes):
                mask = (y == cls)
                fig.add_trace(
                    go.Scatter(
                        x=layer_pca[mask, 0],
                        y=layer_pca[mask, 1],
                        mode='markers',
                        marker=dict(color=colors[j], size=8),
                        name=class_names[j],
                        legendgroup=f"class_{cls}",  # Use consistent legend group
                        showlegend=False  # Don't show duplicate legends
                    ),
                    row=1, col=i+2
                )
        
        # Update layout with a cleaner legend
        fig.update_layout(
            height=500,
            width=250 * n_plots,
            title="PCA Projections Across Network Layers",
            legend_title="Classes",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        for i in range(n_plots):
            fig.update_xaxes(title_text="PC1", row=1, col=i+1)
            fig.update_yaxes(title_text="PC2", row=1, col=i+1)
        
        return fig

    def visualize_tsne_projection(self, X, y, layer_outputs, perplexity=30):
        """
        Visualize t-SNE projections of the data at different layers of the network.
        
        Args:
            X: Input data
            y: Labels
            layer_outputs: Dictionary of layer outputs
            perplexity: Perplexity parameter for t-SNE
        
        Returns:
            Plotly figure with t-SNE projections
        """
        # Create subplots
        n_plots = len(layer_outputs) + 1  # Input data + each layer
        fig = make_subplots(rows=1, cols=n_plots, 
                            subplot_titles=["Input Data"] + list(layer_outputs.keys()))
        
        # Get unique classes and their names
        unique_classes = np.unique(y)
        
        # Ensure we have proper class names
        if hasattr(self, 'label_names') and len(self.label_names) >= len(unique_classes):
            class_names = [self.label_names[int(cls)] for cls in unique_classes]
        else:
            class_names = [f"Class {i}" for i in unique_classes]
        
        # Create a color map for the classes
        colors = px.colors.qualitative.Set1[:len(unique_classes)]
        
        # t-SNE for input data
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Plot input data with proper class labels
        for i, cls in enumerate(unique_classes):
            mask = (y == cls)
            fig.add_trace(
                go.Scatter(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8),
                    name=class_names[i],  # Use proper class name
                    legendgroup=f"class_{cls}",  # Use consistent legend group
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # t-SNE for each layer
        for i, (layer_name, layer_output) in enumerate(layer_outputs.items()):
            # Convert to numpy if tensor
            if torch.is_tensor(layer_output):
                layer_output = layer_output.detach().cpu().numpy()
            
            # Reshape if needed
            if len(layer_output.shape) > 2:
                layer_output = layer_output.reshape(layer_output.shape[0], -1)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            layer_tsne = tsne.fit_transform(layer_output)
            
            # Plot each class with proper labels
            for j, cls in enumerate(unique_classes):
                mask = (y == cls)
                fig.add_trace(
                    go.Scatter(
                        x=layer_tsne[mask, 0],
                        y=layer_tsne[mask, 1],
                        mode='markers',
                        marker=dict(color=colors[j], size=8),
                        name=class_names[j],
                        legendgroup=f"class_{cls}",  # Use consistent legend group
                        showlegend=False  # Don't show duplicate legends
                    ),
                    row=1, col=i+2
                )
        
        # Update layout with a cleaner legend
        fig.update_layout(
            height=500,
            width=250 * n_plots,
            title="t-SNE Projections Across Network Layers",
            legend_title="Classes",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
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