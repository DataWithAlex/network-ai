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

class NetworkVisualizer:
    def __init__(self, model, feature_names, label_names):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        
        # Define colors for different layer types
        self.layer_colors = {
            'input': 'rgba(65, 105, 225, 0.7)',  # Royal blue
            'hidden': 'rgba(50, 205, 50, 0.7)',  # Lime green
            'output': 'rgba(220, 20, 60, 0.7)',  # Crimson
            'conv': 'rgba(65, 105, 225, 0.7)',   # Blue
            'pool': 'rgba(255, 165, 0, 0.7)',    # Orange
            'activation': 'rgba(147, 112, 219, 0.6)',  # Medium purple
            'dropout': 'rgba(169, 169, 169, 0.6)',  # Dark gray
            'batch_norm': 'rgba(255, 215, 0, 0.6)',  # Gold
            'fc': 'rgba(60, 179, 113, 0.7)'      # Medium sea green
        }
        
        # Define shapes for different layer types
        self.layer_shapes = {
            'input': 'rect',
            'hidden': 'rect',
            'output': 'rect',
            'conv': 'rect',
            'pool': 'rect',
            'activation': 'circle',
            'dropout': 'rect',
            'batch_norm': 'rect',
            'fc': 'rect'
        }

    def plot_network_architecture(self):
        """Create a static network visualization using matplotlib"""
        # Extract model structure
        input_size = self.model.input_size
        hidden_sizes = self.model.hidden_sizes
        output_size = self.model.output_size
        
        # Create a graph
        G = nx.DiGraph()
        
        # Add nodes for input layer
        for i in range(input_size):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature {i+1}"
            G.add_node(f"input_{i}", layer="input", label=feature_name, pos=(0, -i))
        
        # Add nodes for hidden layers
        for layer_idx, layer_size in enumerate(hidden_sizes):
            layer_x = layer_idx + 1
            for i in range(layer_size):
                G.add_node(f"hidden_{layer_idx}_{i}", layer=f"hidden_{layer_idx}", 
                           label=f"Neuron {i+1}", pos=(layer_x, -i))
                
                # Add edges from previous layer
                if layer_idx == 0:  # Connect from input layer
                    for j in range(input_size):
                        G.add_edge(f"input_{j}", f"hidden_{layer_idx}_{i}")
                else:  # Connect from previous hidden layer
                    for j in range(hidden_sizes[layer_idx-1]):
                        G.add_edge(f"hidden_{layer_idx-1}_{j}", f"hidden_{layer_idx}_{i}")
        
        # Add nodes for output layer
        for i in range(output_size):
            class_name = self.label_names[i] if i < len(self.label_names) else f"Class {i+1}"
            G.add_node(f"output_{i}", layer="output", label=class_name, 
                      pos=(len(hidden_sizes)+1, -i))
            
            # Connect from last hidden layer
            for j in range(hidden_sizes[-1]):
                G.add_edge(f"hidden_{len(hidden_sizes)-1}_{j}", f"output_{i}")
        
        # Extract positions for visualization
        pos = nx.get_node_attributes(G, 'pos')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes with different colors for each layer
        input_nodes = [node for node, attrs in G.nodes(data=True) if attrs['layer'] == 'input']
        hidden_nodes = [node for node, attrs in G.nodes(data=True) if 'hidden' in attrs['layer']]
        output_nodes = [node for node, attrs in G.nodes(data=True) if attrs['layer'] == 'output']
        
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='royalblue', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='limegreen', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='salmon', node_size=500, alpha=0.8)
        
        # Get the weights and color edges based on weight values
        weights = self.model.get_layer_weights()
        
        # Add a title with hyperparameters
        hyperparams = self.model.get_hyperparameters()
        plt.title(f"Neural Network Architecture\n"
                 f"Input: {hyperparams['Input Size']} | "
                 f"Hidden: {hyperparams['Hidden Layers']} | "
                 f"Output: {hyperparams['Output Size']} | "
                 f"Activation: {hyperparams['Activation Function']}")
        
        # Draw edges with transparency
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        
        # Add labels
        labels = {node: attr['label'] for node, attr in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.axis('off')
        
        return fig

    def plot_network_interactive(self):
        """Create an interactive detailed network visualization"""
        # Extract model structure and weights
        hyperparams = self.model.get_hyperparameters()
        layer_weights = self.model.get_layer_weights()
        
        # For MLPs
        if "MLP" in hyperparams["Model Type"]:
            input_size = hyperparams["Input Size"]
            hidden_sizes = hyperparams["Hidden Layers"]
            output_size = hyperparams["Output Size"]
            dropout_rate = hyperparams["Dropout Rate"]
            activation_fn = hyperparams["Activation Function"]
            
            # Create a list of all layers (linear + activation + dropout)
            all_layers = []
            
            # Add input layer
            all_layers.append({
                'name': 'Input',
                'type': 'input',
                'size': input_size,
                'details': f'Input Features: {input_size}',
                'feature_names': self.feature_names,
                'index': 0
            })
            
            # Add hidden layers with activations and dropout
            layer_index = 1
            for i, size in enumerate(hidden_sizes):
                # Linear layer
                all_layers.append({
                    'name': f'FC-{i+1}',
                    'type': 'fc',
                    'size': size,
                    'details': f'Fully Connected: {size} neurons',
                    'weight_shape': f'({size} x {input_size if i==0 else hidden_sizes[i-1]})',
                    'layer_key': f'linear_{i}',
                    'prev_size': input_size if i==0 else hidden_sizes[i-1],
                    'index': layer_index
                })
                layer_index += 1
                
                # Activation layer
                all_layers.append({
                    'name': f'{activation_fn}-{i+1}',
                    'type': 'activation',
                    'size': size,
                    'details': f'{activation_fn} Activation',
                    'index': layer_index
                })
                layer_index += 1
                
                # Dropout layer (if applicable)
                if dropout_rate > 0:
                    all_layers.append({
                        'name': f'Dropout-{i+1}',
                        'type': 'dropout',
                        'size': size,
                        'details': f'Dropout (p={dropout_rate})',
                        'index': layer_index
                    })
                    layer_index += 1
            
            # Add output layer
            all_layers.append({
                'name': 'Output',
                'type': 'output',
                'size': output_size,
                'details': f'Output Classes: {output_size}',
                'label_names': self.label_names,
                'layer_key': 'linear_output',
                'prev_size': hidden_sizes[-1],
                'index': layer_index
            })
            
            # Create figure
            fig = go.Figure()
            
            # Calculate dimensions for visualization
            max_layer_size = max([layer['size'] for layer in all_layers])
            layer_width = 120  # Base width for all layers
            layer_height_unit = 20  # Height per neuron
            layer_spacing = 200  # Horizontal spacing between layers
            neuron_spacing = 40  # Vertical spacing between neurons in the same layer
            
            # Draw connectors between layers (before drawing shapes to be in background)
            for i in range(len(all_layers)-1):
                current_layer = all_layers[i]
                next_layer = all_layers[i+1]
                
                # For linear layers, draw connections to next layer with weight information
                if current_layer['type'] in ['fc', 'input'] and next_layer['type'] in ['fc', 'activation', 'output']:
                    # Get weight information if available
                    weights_info = None
                    if 'layer_key' in current_layer and current_layer['layer_key'] in layer_weights:
                        weights_info = layer_weights[current_layer['layer_key']]
                    
                    # Determine number of connections to show (max 10 per layer for clarity)
                    current_size = current_layer['size']
                    next_size = next_layer['size']
                    
                    # If too many connections, just show a representative sample with explanation
                    if current_size * next_size > 20:
                        # Calculate start and end positions for representative connections
                        x0 = current_layer['index'] * layer_spacing
                        x1 = next_layer['index'] * layer_spacing
                        
                        # Draw dotted line between centers of layers with explanation
                        fig.add_trace(go.Scatter(
                            x=[x0 + layer_width/2, x1 - layer_width/2],
                            y=[0, 0],
                            mode="lines",
                            line=dict(width=2, color="rgba(100, 100, 100, 0.8)", dash="dot"),
                            text=f"{current_size} x {next_size} = {current_size * next_size} connections",
                            hoverinfo="text",
                            showlegend=False
                        ))
                        
                        # Add annotation about connections
                        fig.add_annotation(
                            x=(x0 + x1)/2,
                            y=0,
                            text=f"{current_size}×{next_size}",
                            showarrow=False,
                            font=dict(size=10, color="rgba(100, 100, 100, 0.8)"),
                            bgcolor="rgba(255, 255, 255, 0.7)"
                        )
                    else:
                        # Draw actual connections between individual neurons
                        for j in range(min(current_size, 5)):
                            y0 = (j - min(current_size, 10)/2 + 0.5) * neuron_spacing
                            
                            for k in range(min(next_size, 5)):
                                y1 = (k - min(next_size, 10)/2 + 0.5) * neuron_spacing
                                
                                # Determine connection weight if available
                                weight_value = None
                                if weights_info is not None:
                                    if j < weights_info['weight'].shape[1] and k < weights_info['weight'].shape[0]:
                                        weight_value = weights_info['weight'][k, j]
                                
                                # Determine line color based on weight
                                line_color = "rgba(100, 100, 100, 0.3)"
                                line_width = 1
                                if weight_value is not None:
                                    # Normalize weight to determine color intensity
                                    weight_abs = abs(weight_value)
                                    alpha = min(0.8, 0.2 + weight_abs * 0.6)
                                    if weight_value > 0:
                                        line_color = f"rgba(0, 100, 0, {alpha})"  # Green for positive
                                    else:
                                        line_color = f"rgba(220, 0, 0, {alpha})"  # Red for negative
                                    
                                    # Adjust line width based on weight magnitude
                                    line_width = 1 + 2 * min(1, weight_abs)
                                
                                # Draw connection
                                fig.add_trace(go.Scatter(
                                    x=[current_layer['index'] * layer_spacing + layer_width/2, 
                                       next_layer['index'] * layer_spacing - layer_width/2],
                                    y=[y0, y1],
                                    mode="lines",
                                    line=dict(width=line_width, color=line_color),
                                    hoverinfo="text",
                                    text=f"Weight: {weight_value:.4f}" if weight_value is not None else "Connection",
                                    showlegend=False
                                ))
            
            # Draw explanatory annotations
            # Annotate data flow
            fig.add_annotation(
                x=(all_layers[0]['index'] * layer_spacing + all_layers[1]['index'] * layer_spacing) / 2,
                y=max_layer_size * neuron_spacing / 2 + 50,
                text="Data flow →",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(100, 100, 100, 0.7)",
                ax=50,
                ay=0,
                xref="x",
                yref="y"
            )
            
            # Annotate weights
            fig.add_annotation(
                x=(all_layers[0]['index'] * layer_spacing + all_layers[1]['index'] * layer_spacing) / 2,
                y=max_layer_size * neuron_spacing / 2 - 50,
                text="Weights transform inputs",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(100, 100, 100, 0.7)",
                ax=-20,
                ay=-20,
                xref="x",
                yref="y"
            )
            
            # Draw each layer
            for i, layer in enumerate(all_layers):
                layer_type = layer['type']
                layer_size = layer['size']
                layer_name = layer['name']
                
                # Determine layer color
                color = self.layer_colors.get(layer_type, 'rgba(100, 100, 100, 0.7)')
                
                # Determine shape type
                shape_type = self.layer_shapes.get(layer_type, 'rect')
                
                # Calculate layer dimensions and position
                layer_height = max(100, min(layer_size * layer_height_unit, 500))  # Constrain height
                x_pos = layer['index'] * layer_spacing
                
                # Draw the main layer shape
                if layer_size > 10:
                    # For large layers, draw a rectangle with summary and sample neurons
                    # Main rectangle
                    fig.add_shape(
                        type="rect",
                        x0=x_pos - layer_width/2,
                        y0=-layer_height/2,
                        x1=x_pos + layer_width/2,
                        y1=layer_height/2,
                        fillcolor=color,
                        line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                    )
                    
                    # Layer name
                    fig.add_annotation(
                        x=x_pos,
                        y=0,
                        text=f"{layer_name}<br>{layer_size}",
                        showarrow=False,
                        font=dict(color="white", size=10)
                    )
                    
                    # Draw sample neurons (max 5) at edges to show size
                    samples = min(5, layer_size)
                    for j in range(samples):
                        y_offset = (j - samples/2 + 0.5) * neuron_spacing
                        # Small circle to represent a neuron
                        fig.add_shape(
                            type="circle",
                            x0=x_pos - 10,
                            y0=y_offset - 5,
                            x1=x_pos + 10,
                            y1=y_offset + 5,
                            fillcolor="rgba(255, 255, 255, 0.7)",
                            line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                        )
                    
                    # Add "..." to show there are more neurons
                    if layer_size > samples:
                        fig.add_annotation(
                            x=x_pos,
                            y=(samples/2 + 0.5) * neuron_spacing + 20,
                            text="...",
                            showarrow=False,
                            font=dict(size=16)
                        )
                
                else:
                    # For smaller layers, show individual neurons
                    for j in range(layer_size):
                        y_pos = (j - layer_size/2 + 0.5) * neuron_spacing
                        
                        # Draw each neuron
                        if shape_type == "circle":
                            # For activation functions
                            fig.add_shape(
                                type="circle",
                                x0=x_pos - layer_width/4,
                                y0=y_pos - 15,
                                x1=x_pos + layer_width/4,
                                y1=y_pos + 15,
                                fillcolor=color,
                                line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                            )
                        else:
                            # For other layer types
                            fig.add_shape(
                                type="rect",
                                x0=x_pos - layer_width/2,
                                y0=y_pos - 15,
                                x1=x_pos + layer_width/2,
                                y1=y_pos + 15,
                                fillcolor=color,
                                line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                            )
                        
                        # Add label for input/output neurons
                        if layer_type == "input" and j < len(self.feature_names):
                            fig.add_annotation(
                                x=x_pos,
                                y=y_pos,
                                text=f"{self.feature_names[j][:10]}",
                                showarrow=False,
                                font=dict(color="white", size=8)
                            )
                        elif layer_type == "output" and j < len(self.label_names):
                            fig.add_annotation(
                                x=x_pos,
                                y=y_pos,
                                text=f"{self.label_names[j][:10]}",
                                showarrow=False,
                                font=dict(color="white", size=8)
                            )
                
                # Add detailed hover information
                hover_text = layer['details']
                if layer_type == 'fc' and 'layer_key' in layer and layer['layer_key'] in layer_weights:
                    weight_data = layer_weights[layer['layer_key']]
                    weights = weight_data['weight']
                    biases = weight_data['bias']
                    
                    avg_weight = np.mean(np.abs(weights))
                    max_weight = np.max(np.abs(weights))
                    avg_bias = np.mean(np.abs(biases)) if biases is not None else 0
                    
                    hover_text += f"<br>Avg Weight: {avg_weight:.4f}"
                    hover_text += f"<br>Max Weight: {max_weight:.4f}"
                    hover_text += f"<br>Avg Bias: {avg_bias:.4f}"
                    hover_text += f"<br>Shape: {weights.shape}"
                    
                    # Add info on weight distribution
                    pos_weights = np.sum(weights > 0)
                    neg_weights = np.sum(weights < 0)
                    hover_text += f"<br>Positive weights: {pos_weights}"
                    hover_text += f"<br>Negative weights: {neg_weights}"
                
                if 'feature_names' in layer and isinstance(layer['feature_names'], (list, tuple, np.ndarray)) and len(layer['feature_names']) > 0:
                    hover_text += f"<br>Features: {', '.join(layer['feature_names'][:5])}"
                    if len(layer['feature_names']) > 5:
                        hover_text += f" + {len(layer['feature_names'])-5} more"
                
                if 'label_names' in layer and isinstance(layer['label_names'], (list, tuple, np.ndarray)) and len(layer['label_names']) > 0:
                    hover_text += f"<br>Classes: {', '.join(layer['label_names'][:5])}"
                    if len(layer['label_names']) > 5:
                        hover_text += f" + {len(layer['label_names'])-5} more"
                
                # Add invisible scatter point for hover
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[0],
                    mode="markers",
                    marker=dict(
                        size=layer_height,
                        color="rgba(0, 0, 0, 0)",
                        line=dict(width=0)
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    showlegend=False
                ))
                
                # Add dotted connection to show data flow for activation and dropout layers
                if layer_type in ['activation', 'dropout'] and i > 0:
                    prev_x = all_layers[i-1]['index'] * layer_spacing
                    fig.add_shape(
                        type="line",
                        x0=prev_x + layer_width/2,
                        y0=0,
                        x1=x_pos - layer_width/2,
                        y1=0,
                        line=dict(
                            color="rgba(100, 100, 100, 0.5)",
                            width=1,
                            dash="dot"
                        )
                    )
            
            # Add legend for layer types
            legend_items = {
                'input': 'Input Layer',
                'fc': 'Fully Connected Layer',
                'activation': 'Activation Function',
                'dropout': 'Dropout Layer',
                'output': 'Output Layer'
            }
            
            # Only include layer types that are actually used
            used_layer_types = set(layer['type'] for layer in all_layers)
            
            # Add legend items
            legend_x = 0.02
            legend_y = 1.15
            for layer_type, layer_name in legend_items.items():
                if layer_type in used_layer_types:
                    fig.add_shape(
                        type=self.layer_shapes.get(layer_type, 'rect'),
                        x0=legend_x,
                        y0=legend_y - 0.03,
                        x1=legend_x + 0.02,
                        y1=legend_y,
                        fillcolor=self.layer_colors[layer_type],
                        line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                        xref="paper",
                        yref="paper"
                    )
                    
                    fig.add_annotation(
                        x=legend_x + 0.06,
                        y=legend_y - 0.015,
                        text=layer_name,
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        align="left"
                    )
                    
                    legend_x += 0.2
                    if legend_x > 0.8:
                        legend_x = 0.02
                        legend_y -= 0.05
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f"{hyperparams['Model Type']} Architecture",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                width=1000,
                height=600,
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1
                ),
                margin=dict(t=120, b=20, l=20, r=20),
                hovermode="closest"
            )
            
            # Add title with model summary
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Input: {input_size} | Hidden: {hidden_sizes} | Output: {output_size} | Activation: {activation_fn} | Dropout: {dropout_rate}",
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
            
            # Add parameter count annotation
            fig.add_annotation(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Total Parameters: {hyperparams['Total Parameters']} | Trainable: {hyperparams['Trainable Parameters']}",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
            
            return fig
        else:
            # Placeholder for other model types (like CNNs)
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Visualization for {hyperparams['Model Type']} is not implemented yet",
                showarrow=False,
                font=dict(size=20)
            )
            return fig

    def visualize_feature_importance(self):
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
    
    def visualize_decision_boundary(self, X, y, resolution=0.01):
        """Visualize decision boundary for 2D data"""
        # This only works for 2 features
        if X.shape[1] != 2:
            return None
            
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