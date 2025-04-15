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
        """Create an interactive layered network visualization similar to VGG diagram"""
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
                'feature_names': self.feature_names
            })
            
            # Add hidden layers with activations and dropout
            for i, size in enumerate(hidden_sizes):
                # Linear layer
                all_layers.append({
                    'name': f'FC-{i+1}',
                    'type': 'fc',
                    'size': size,
                    'details': f'Fully Connected: {size} neurons',
                    'weight_shape': f'({size} x {input_size if i==0 else hidden_sizes[i-1]})'
                })
                
                # Activation layer
                all_layers.append({
                    'name': f'{activation_fn}-{i+1}',
                    'type': 'activation',
                    'size': size,
                    'details': f'{activation_fn} Activation'
                })
                
                # Dropout layer (if applicable)
                if dropout_rate > 0:
                    all_layers.append({
                        'name': f'Dropout-{i+1}',
                        'type': 'dropout',
                        'size': size,
                        'details': f'Dropout (p={dropout_rate})'
                    })
            
            # Add output layer
            all_layers.append({
                'name': 'Output',
                'type': 'output',
                'size': output_size,
                'details': f'Output Classes: {output_size}',
                'label_names': self.label_names
            })
            
            # Create figure
            fig = go.Figure()
            
            # Calculate dimensions for visualization
            max_layer_size = max([layer['size'] for layer in all_layers])
            layer_width = 100  # Base width for all layers
            layer_spacing = 150  # Horizontal spacing between layers
            neuron_spacing = 40  # Vertical spacing between neurons in the same layer
            
            # Draw each layer
            for i, layer in enumerate(all_layers):
                layer_type = layer['type']
                layer_size = layer['size']
                layer_name = layer['name']
                
                # Determine layer color
                color = self.layer_colors.get(layer_type, 'rgba(100, 100, 100, 0.7)')
                
                # Calculate layer dimensions
                layer_height = layer_size * 20  # Adjust based on neurons
                
                # Calculate position
                x_pos = i * layer_spacing
                
                # For layers with many neurons, use a rectangle with annotations
                if layer_size > 10:
                    # Draw rectangle for the layer
                    fig.add_shape(
                        type="rect",
                        x0=x_pos - layer_width/2,
                        y0=-layer_height/2,
                        x1=x_pos + layer_width/2,
                        y1=layer_height/2,
                        fillcolor=color,
                        line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                    )
                    
                    # Add text for layer name and size
                    fig.add_annotation(
                        x=x_pos,
                        y=0,
                        text=f"{layer_name}<br>{layer_size}",
                        showarrow=False,
                        font=dict(color="white", size=10)
                    )
                    
                    # Add detailed info for hover
                    hover_text = layer['details']
                    if 'feature_names' in layer and layer['feature_names']:
                        hover_text += f"<br>Features: {', '.join(layer['feature_names'][:5])}"
                        if len(layer['feature_names']) > 5:
                            hover_text += f" + {len(layer['feature_names'])-5} more"
                    
                    if 'label_names' in layer and layer['label_names']:
                        hover_text += f"<br>Classes: {', '.join(layer['label_names'][:5])}"
                        if len(layer['label_names']) > 5:
                            hover_text += f" + {len(layer['label_names'])-5} more"
                    
                    if 'weight_shape' in layer:
                        hover_text += f"<br>Weights: {layer['weight_shape']}"
                    
                    # Add invisible scatter point for hover information
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[0],
                        mode="markers",
                        marker=dict(
                            size=layer_width,
                            color="rgba(0, 0, 0, 0)",
                            line=dict(width=0)
                        ),
                        text=hover_text,
                        hoverinfo="text",
                        showlegend=False
                    ))
                else:
                    # For smaller layers, show individual neurons
                    for j in range(layer_size):
                        y_pos = (j - layer_size/2 + 0.5) * neuron_spacing
                        
                        # Draw the neuron
                        fig.add_shape(
                            type="rect" if layer_type in ["fc", "input", "output"] else "circle",
                            x0=x_pos - layer_width/4,
                            y0=y_pos - 10,
                            x1=x_pos + layer_width/4,
                            y1=y_pos + 10,
                            fillcolor=color,
                            line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                        )
                        
                        # Add name annotation for special neurons
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
                        
                        # Add hover information
                        hover_text = layer['details']
                        if layer_type == "input" and j < len(self.feature_names):
                            hover_text = f"Feature: {self.feature_names[j]}"
                        elif layer_type == "output" and j < len(self.label_names):
                            hover_text = f"Class: {self.label_names[j]}"
                        
                        fig.add_trace(go.Scatter(
                            x=[x_pos],
                            y=[y_pos],
                            mode="markers",
                            marker=dict(
                                size=20,
                                color="rgba(0, 0, 0, 0)",
                                line=dict(width=0)
                            ),
                            text=hover_text,
                            hoverinfo="text",
                            showlegend=False
                        ))
            
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
                        type="rect",
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
                width=900,
                height=500,
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
                margin=dict(t=100, b=20, l=20, r=20),
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