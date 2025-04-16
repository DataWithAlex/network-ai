import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

class InteractiveNetworkVisualizer:
    """
    Class for creating interactive network visualizations using Plotly
    """
    def __init__(self, model, feature_names, label_names):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        
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
                    'name': f'Activation-{i+1}',
                    'type': 'activation',
                    'size': size,
                    'details': f'{activation_fn}: {size} neurons',
                    'index': layer_index
                })
                layer_index += 1
                
                # Dropout layer (if used)
                if dropout_rate > 0:
                    all_layers.append({
                        'name': f'Dropout-{i+1}',
                        'type': 'dropout',
                        'size': size,
                        'details': f'Dropout: {dropout_rate}',
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
                'prev_size': hidden_sizes[-1],
                'layer_key': 'linear_output',
                'index': layer_index
            })
            
            # Make sure all layer types are mapped to a color and shape by adding fallbacks
            for layer in all_layers:
                layer_type = layer['type']
                if layer_type not in self.layer_colors:
                    # Use hidden layer color/shape for unknown layer types
                    self.layer_colors[layer_type] = self.layer_colors['hidden']
                    self.layer_shapes[layer_type] = self.layer_shapes['hidden']
            
            # Create figure
            fig = go.Figure()
            
            # Set layout dimensions based on largest layer
            max_layer_size = max([layer['size'] for layer in all_layers])
            layer_height_unit = 40  # pixels per neuron
            
            # Spacing and size parameters
            layer_spacing = 200  # horizontal spacing between layers
            max_neurons_to_show = 10  # max neurons to show per layer
            neuron_spacing = 30  # vertical spacing between neurons
            
            # Create network visualization
            # Loop through each layer
            for layer_idx, layer in enumerate(all_layers):
                layer_type = layer['type']
                layer_x = layer_idx * layer_spacing
                
                # Draw layer box
                layer_height = layer['size'] * layer_height_unit
                layer_width = 120
                
                # Calculate y position to center the layer
                layer_y = -(layer_height / 2)
                
                # Add layer shape
                fig.add_shape(
                    type=self.layer_shapes.get(layer_type, 'rect'),
                    x0=layer_x,
                    y0=layer_y,
                    x1=layer_x + layer_width,
                    y1=layer_y + layer_height,
                    fillcolor=self.layer_colors.get(layer_type, self.layer_colors['hidden']),
                    line=dict(color="rgba(50, 50, 50, 0.8)", width=1)
                )
                
                # Add layer title
                fig.add_annotation(
                    x=layer_x + layer_width/2,
                    y=layer_y - 30,
                    text=layer['name'],
                    showarrow=False,
                    font=dict(size=12, color="black")
                )
                
                # Add layer details
                fig.add_annotation(
                    x=layer_x + layer_width/2,
                    y=layer_y - 15,
                    text=layer['details'],
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
                
                # Draw connections to previous layer (for fc and output layers)
                if layer_type in ['fc', 'output'] and layer_idx > 0:
                    prev_layer = all_layers[layer_idx-1]
                    # If previous layer is activation or dropout, find the last fc layer
                    if prev_layer['type'] in ['activation', 'dropout']:
                        for j in range(layer_idx-1, -1, -1):
                            if all_layers[j]['type'] in ['fc', 'input']:
                                prev_layer = all_layers[j]
                                break
                    
                    prev_layer_x = prev_layer['index'] * layer_spacing
                    
                    # Get weights for this layer if available
                    if 'layer_key' in layer and layer['layer_key'] in layer_weights:
                        weights = layer_weights[layer['layer_key']]['weight']
                    else:
                        # Create dummy weights if not available (for visualization only)
                        weights = np.random.randn(layer['size'], prev_layer['size']) * 0.1
                    
                    # Draw connections between neurons
                    # If layers are large, only show connections for a subset of neurons
                    show_neurons = min(max_neurons_to_show, layer['size'])
                    prev_show_neurons = min(max_neurons_to_show, prev_layer['size'])
                    
                    # Determine which neurons to show (evenly spaced)
                    if layer['size'] > show_neurons:
                        neuron_indices = np.linspace(0, layer['size']-1, show_neurons, dtype=int)
                    else:
                        neuron_indices = np.arange(layer['size'])
                        
                    if prev_layer['size'] > prev_show_neurons:
                        prev_neuron_indices = np.linspace(0, prev_layer['size']-1, prev_show_neurons, dtype=int)
                    else:
                        prev_neuron_indices = np.arange(prev_layer['size'])
                        
                    # Draw connections
                    for i in neuron_indices:
                        y_pos = (i - layer['size']/2) * neuron_spacing
                        
                        for j in prev_neuron_indices:
                            prev_y_pos = (j - prev_layer['size']/2) * neuron_spacing
                            
                            # Get the weight value for this connection
                            weight_val = weights[i, j] if i < weights.shape[0] and j < weights.shape[1] else 0
                            
                            # Determine line color and width based on weight
                            if weight_val > 0:
                                color = self.weight_pos_color
                            else:
                                color = self.weight_neg_color
                                
                            # Scale line width by weight magnitude (with limits)
                            width = min(5, max(0.5, abs(weight_val) * 2))
                            
                            # Add the connection line
                            fig.add_shape(
                                type="line",
                                x0=prev_layer_x + layer_width,
                                y0=prev_y_pos,
                                x1=layer_x,
                                y1=y_pos,
                                line=dict(color=color, width=width)
                            )
                
                # Draw neurons for this layer
                if layer['size'] > max_neurons_to_show:
                    # For large layers, show dots for a subset of neurons
                    show_neurons = max_neurons_to_show
                    
                    # Show neurons from the middle of the layer
                    for i in range(-show_neurons//2, show_neurons//2 + show_neurons%2):
                        y_pos = i * neuron_spacing
                        
                        # Add neuron circle
                        fig.add_shape(
                            type="circle",
                            x0=layer_x + layer_width/4,
                            y0=y_pos - 10,
                            x1=layer_x + 3*layer_width/4,
                            y1=y_pos + 10,
                            fillcolor="rgba(255, 255, 255, 0.8)",
                            line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                            hoverinfo='text',
                            hoverlabel=dict(bgcolor="white"),
                            hovertext=f"{layer['name']} - Neuron {i+1}"
                        )
                    
                    # Add ellipsis to show there are more
                    fig.add_annotation(
                        x=layer_x + layer_width/2,
                        y=(show_neurons/2 + 1) * neuron_spacing,
                        text="...",
                        showarrow=False,
                        font=dict(size=16, color="black")
                    )
                    
                    fig.add_annotation(
                        x=layer_x + layer_width/2,
                        y=-(show_neurons/2 + 1) * neuron_spacing,
                        text="...",
                        showarrow=False,
                        font=dict(size=16, color="black")
                    )
                else:
                    # For smaller layers, show all neurons
                    for i in range(layer['size']):
                        y_pos = (i - layer['size']/2) * neuron_spacing
                        
                        # Add neuron circle
                        fig.add_shape(
                            type="circle",
                            x0=layer_x + layer_width/4,
                            y0=y_pos - 10,
                            x1=layer_x + 3*layer_width/4,
                            y1=y_pos + 10,
                            fillcolor="rgba(255, 255, 255, 0.8)",
                            line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                            hoverinfo='text',
                            hoverlabel=dict(bgcolor="white"),
                            hovertext=f"{layer['name']} - Neuron {i+1}"
                        )
                        
                        # Add label for input or output layers
                        if layer_type == 'input' and 'feature_names' in layer:
                            label = layer['feature_names'][i] if i < len(layer['feature_names']) else f"Feature {i}"
                            fig.add_annotation(
                                x=layer_x - 5,
                                y=y_pos,
                                text=label,
                                showarrow=False,
                                xanchor="right",
                                font=dict(size=8, color="black")
                            )
                        elif layer_type == 'output' and 'label_names' in layer:
                            label = layer['label_names'][i] if i < len(layer['label_names']) else f"Class {i}"
                            fig.add_annotation(
                                x=layer_x + layer_width + 5,
                                y=y_pos,
                                text=label,
                                showarrow=False,
                                xanchor="left",
                                font=dict(size=8, color="black")
                            )
            
            # Draw explanatory annotations
            # Add legend
            legend_y = max_layer_size * layer_height_unit/2 + 100
            legend_x = 10
            legend_spacing = 30
            
            fig.add_annotation(
                x=legend_x,
                y=legend_y,
                text="Layer Types:",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="black", family="Arial")
            )
            
            # Legend items for each layer type
            legend_items = [
                ('input', 'Input Layer'),
                ('fc', 'Fully Connected'),
                ('activation', 'Activation Function'),
                ('dropout', 'Dropout'),
                ('output', 'Output Layer')
            ]
            
            for i, (layer_type, description) in enumerate(legend_items):
                # Add colored shape
                fig.add_shape(
                    type=self.layer_shapes.get(layer_type, 'rect'),
                    x0=legend_x,
                    y0=legend_y - (i+1)*legend_spacing - 0.03,
                    x1=legend_x + 0.02,
                    y1=legend_y - (i+1)*legend_spacing,
                    fillcolor=self.layer_colors.get(layer_type, self.layer_colors['hidden']),
                    line=dict(color="rgba(50, 50, 50, 0.8)", width=1),
                    xref="paper",
                    yref="paper"
                )
                
                # Add text description
                fig.add_annotation(
                    x=legend_x + 0.03,
                    y=legend_y - (i+1)*legend_spacing,
                    text=description,
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=10, color="black"),
                    xref="paper",
                    yref="paper"
                )
            
            # Add legend for connection weights
            fig.add_annotation(
                x=legend_x + 0.15,
                y=legend_y,
                text="Connection Weights:",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="black", family="Arial"),
                xref="paper",
                yref="paper"
            )
            
            # Weight legend
            weight_legend = [
                (self.weight_pos_color, "Positive Weight"),
                (self.weight_neg_color, "Negative Weight")
            ]
            
            for i, (color, description) in enumerate(weight_legend):
                # Add line
                fig.add_shape(
                    type="line",
                    x0=legend_x + 0.15,
                    y0=legend_y - (i+1)*legend_spacing,
                    x1=legend_x + 0.17,
                    y1=legend_y - (i+1)*legend_spacing,
                    line=dict(color=color, width=3),
                    xref="paper",
                    yref="paper"
                )
                
                # Add text description
                fig.add_annotation(
                    x=legend_x + 0.18,
                    y=legend_y - (i+1)*legend_spacing,
                    text=description,
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=10, color="black"),
                    xref="paper",
                    yref="paper"
                )
            
            # Layout configuration
            fig.update_layout(
                title=f"Neural Network Architecture: {hyperparams['Model Type']}",
                showlegend=True,
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                width=max(800, (len(all_layers) + 1) * layer_spacing),
                height=max(600, max_layer_size * layer_height_unit + 300),
                margin=dict(l=50, r=50, b=50, t=80),
                hovermode='closest',
                hoverdistance=10,
                hoverlabel=dict(bgcolor="white", font_size=12),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-100, len(all_layers) * layer_spacing + 100]
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                    range=[-max_layer_size * layer_height_unit/2 - 100, max_layer_size * layer_height_unit/2 + 100]
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