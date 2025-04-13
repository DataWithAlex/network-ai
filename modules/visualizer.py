import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
import torch

class NetworkVisualizer:
    def __init__(self, model, feature_names, label_names):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names

    def plot_network_architecture(self):
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
        node_idx = 0
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
                       pos=(len(hidden_sizes) + 1, -i))
            
            # Add edges from last hidden layer
            for j in range(hidden_sizes[-1]):
                G.add_edge(f"hidden_{len(hidden_sizes)-1}_{j}", f"output_{i}")
        
        # Create the figure
        fig = plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Separate nodes by layer for coloring
        input_nodes = [node for node, attr in G.nodes(data=True) if attr['layer'] == 'input']
        hidden_nodes = [node for node, attr in G.nodes(data=True) if 'hidden' in attr['layer']]
        output_nodes = [node for node, attr in G.nodes(data=True) if attr['layer'] == 'output']
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='skyblue', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightgreen', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='salmon', node_size=500, alpha=0.8)
        
        # Draw edges with transparency
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        
        # Add labels
        labels = {node: attr['label'] for node, attr in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title('Neural Network Architecture')
        plt.axis('off')
        
        return fig
    
    def plot_network_interactive(self):
        # Extract model structure
        input_size = self.model.input_size
        hidden_sizes = self.model.hidden_sizes
        output_size = self.model.output_size
        
        # Get weights
        layer_weights = self.model.get_layer_weights()
        
        # Create node positions
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        num_layers = len(layer_sizes)
        
        # Create nodes data
        nodes_x = []
        nodes_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        layer_markers = []
        
        # Layout parameters
        horizontal_spacing = 1 / (num_layers - 1) if num_layers > 1 else 0.5
        
        # Process each layer
        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_x = layer_idx * horizontal_spacing
            
            # Determine node labels and colors based on layer type
            if layer_idx == 0:  # Input layer
                color = 'rgba(65, 105, 225, 0.8)'  # Royal blue
                marker_symbol = 'circle'
                size = 15
                labels = self.feature_names if self.feature_names else [f"Feature {i+1}" for i in range(layer_size)]
            elif layer_idx == num_layers - 1:  # Output layer
                color = 'rgba(220, 20, 60, 0.8)'  # Crimson
                marker_symbol = 'circle'
                size = 15
                labels = [f"Class {i+1}" for i in range(layer_size)]
                if self.label_names is not None and len(self.label_names) > 0:
                    labels = [self.label_names[i] if i < len(self.label_names) else f"Class {i+1}" for i in range(layer_size)]
            else:  # Hidden layers
                color = 'rgba(50, 205, 50, 0.8)'  # Lime green
                marker_symbol = 'circle'
                size = 12
                labels = [f"Neuron {i+1}" for i in range(layer_size)]
            
            # Compute vertical positioning
            if layer_size == 1:
                layer_y = [0.5]
            else:
                layer_y = np.linspace(0, 1, layer_size)
            
            # Add nodes for this layer
            for i, y in enumerate(layer_y):
                nodes_x.append(layer_x)
                nodes_y.append(y)
                node_text.append(f"{labels[i]}")
                node_colors.append(color)
                node_sizes.append(size)
                layer_markers.append(marker_symbol)
        
        # Create edges data
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        
        # Helper to get node index
        def get_node_idx(layer, neuron):
            offset = sum(layer_sizes[:layer])
            return offset + neuron
        
        # Process weights for edges
        node_idx = 0
        layer_idx = 0
        
        for i, (layer_name, layer_data) in enumerate(layer_weights.items()):
            weights = layer_data['weight']
            source_layer = layer_idx
            target_layer = layer_idx + 1
            
            # Normalize weights for visualization
            max_weight = np.max(np.abs(weights))
            for target_neuron, source_weights in enumerate(weights):
                for source_neuron, weight in enumerate(source_weights):
                    source_idx = get_node_idx(source_layer, source_neuron)
                    target_idx = get_node_idx(target_layer, target_neuron)
                    
                    edge_x.extend([nodes_x[source_idx], nodes_x[target_idx], None])
                    edge_y.extend([nodes_y[source_idx], nodes_y[target_idx], None])
                    
                    # Color based on weight sign (red for negative, blue for positive)
                    normalized_weight = weight / max_weight if max_weight > 0 else 0
                    if weight > 0:
                        edge_color = f'rgba(0, 0, 255, {min(abs(normalized_weight), 1) * 0.7})'  # Blue
                    else:
                        edge_color = f'rgba(255, 0, 0, {min(abs(normalized_weight), 1) * 0.7})'  # Red
                    
                    edge_colors.extend([edge_color, edge_color, edge_color])
                    edge_widths.extend([2 * abs(normalized_weight), 2 * abs(normalized_weight), 2 * abs(normalized_weight)])
            
            layer_idx += 1
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges
        for i in range(0, len(edge_x), 3):
            fig.add_trace(go.Scatter(
                x=edge_x[i:i+3],
                y=edge_y[i:i+3],
                mode='lines',
                line=dict(color=edge_colors[i], width=edge_widths[i]),
                hoverinfo='none'
            ))
        
        # Add nodes
        for i, (x, y, text, color, size, marker) in enumerate(zip(nodes_x, nodes_y, node_text, node_colors, node_sizes, layer_markers)):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    symbol=marker,
                    size=size,
                    color=color,
                    line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
                ),
                name=text,
                text=text,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Neural Network Architecture',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
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
    
    def visualize_feature_importance(self):
        """Visualize feature importance based on first layer weights"""
        weights = self.model.get_layer_weights()
        first_layer_weights = weights['linear_0']['weight'].T  # [input_features, hidden_neurons]
        
        # Compute absolute importance (magnitude of weights)
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)[::-1]
        sorted_features = [self.feature_names[i] if i < len(self.feature_names) 
                          else f"Feature {i+1}" for i in sorted_idx]
        sorted_importance = importance[sorted_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_features)), sorted_importance, align='center')
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()  # Display the most important at the top
        ax.set_xlabel('Average Absolute Weight')
        ax.set_title('Feature Importance based on First Layer Weights')
        
        return fig
    
    def visualize_pca_projection(self, X, y, layer_outputs=None):
        """Visualize the original data and layer outputs using PCA projection"""
        # Prepare the data
        if layer_outputs is None:
            # If no layer outputs are provided, use the original data
            data_dict = {'Original Data': X}
        else:
            # Add original data and layer outputs
            data_dict = {'Original Data': X}
            for layer_name, output in layer_outputs.items():
                data_dict[layer_name] = output.squeeze().detach().numpy()
        
        # Create figure with subplots
        n_plots = len(data_dict)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # If only one plot, wrap in list
        if n_plots == 1:
            axes = [axes]
        
        # Plot each dataset
        for i, (name, data) in enumerate(data_dict.items()):
            # Apply PCA
            pca = PCA(n_components=2)
            if data.ndim == 1:
                data = data.reshape(1, -1)  # Ensure 2D
            projected_data = pca.fit_transform(data)
            
            # Create scatter plot
            scatter = axes[i].scatter(
                projected_data[:, 0], 
                projected_data[:, 1], 
                c=y, 
                cmap='viridis', 
                alpha=0.7
            )
            
            # Add labels and title
            axes[i].set_xlabel('Principal Component 1')
            axes[i].set_ylabel('Principal Component 2')
            axes[i].set_title(f'PCA Projection: {name}')
            
            # Add legend
            if i == 0:  # Only add legend to first plot
                legend1 = axes[i].legend(*scatter.legend_elements(),
                                        loc="upper right", title="Classes")
                axes[i].add_artist(legend1)
        
        plt.tight_layout()
        return fig
    
    def visualize_tsne_projection(self, X, y, layer_outputs=None, perplexity=30):
        """Visualize the original data and layer outputs using t-SNE projection"""
        # Prepare the data
        if layer_outputs is None:
            # If no layer outputs are provided, use the original data
            data_dict = {'Original Data': X}
        else:
            # Add original data and layer outputs
            data_dict = {'Original Data': X}
            for layer_name, output in layer_outputs.items():
                data_dict[layer_name] = output.squeeze().detach().numpy()
        
        # Create figure with subplots
        n_plots = len(data_dict)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # If only one plot, wrap in list
        if n_plots == 1:
            axes = [axes]
        
        # Plot each dataset
        for i, (name, data) in enumerate(data_dict.items()):
            # Apply t-SNE
            if data.shape[0] > 1:  # t-SNE needs at least 2 samples
                tsne = TSNE(n_components=2, perplexity=min(perplexity, data.shape[0]-1), random_state=42)
                if data.ndim == 1:
                    data = data.reshape(1, -1)  # Ensure 2D
                projected_data = tsne.fit_transform(data)
                
                # Create scatter plot
                scatter = axes[i].scatter(
                    projected_data[:, 0], 
                    projected_data[:, 1], 
                    c=y, 
                    cmap='viridis', 
                    alpha=0.7
                )
                
                # Add legend
                if i == 0:  # Only add legend to first plot
                    legend1 = axes[i].legend(*scatter.legend_elements(),
                                            loc="upper right", title="Classes")
                    axes[i].add_artist(legend1)
            else:
                axes[i].text(0.5, 0.5, "Not enough samples for t-SNE", 
                            ha='center', va='center', fontsize=12)
            
            # Add labels and title
            axes[i].set_xlabel('t-SNE Feature 1')
            axes[i].set_ylabel('t-SNE Feature 2')
            axes[i].set_title(f't-SNE Projection: {name}')
        
        plt.tight_layout()
        return fig 