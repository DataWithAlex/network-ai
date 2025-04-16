import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import colorsys

class ActivationTracer:
    """
    Specialized class for tracing activations through neural networks
    Works with any model that implements get_intermediate_features()
    """
    def __init__(self, model, feature_names, label_names):
        self.model = model
        self.feature_names = feature_names
        self.label_names = label_names
        
    def trace_sample(self, sample_input, sample_label=None):
        """
        Trace a sample input through the network and return activations at each layer
        
        Args:
            sample_input: Input tensor or array
            sample_label: Optional ground truth label
            
        Returns:
            Dict containing activations and prediction information
        """
        # Ensure sample input is a tensor and add batch dimension if needed
        if not torch.is_tensor(sample_input):
            sample_input = torch.tensor(sample_input, dtype=torch.float32)
        if len(sample_input.shape) == 1:
            sample_input_batch = sample_input.unsqueeze(0)
        else:
            sample_input_batch = sample_input
            
        # Get activations and prediction
        with torch.no_grad():
            activations = self.model.get_intermediate_features(sample_input_batch)
            sample_output = self.model(sample_input_batch)
            predicted_class = torch.argmax(sample_output, dim=1).item()
            probabilities = torch.softmax(sample_output, dim=1)[0].tolist()
        
        # Get hyperparameters to understand the network structure
        hyperparams = self.model.get_hyperparameters()
        layer_weights = self.model.get_layer_weights()
        
        # Return comprehensive activation information
        return {
            "input": sample_input.detach().numpy() if torch.is_tensor(sample_input) else sample_input,
            "activations": activations,
            "output": sample_output.detach().numpy(),
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "predicted_label": self.label_names[predicted_class] if predicted_class < len(self.label_names) else f"Class {predicted_class}",
            "true_label": self.label_names[sample_label] if sample_label is not None and sample_label < len(self.label_names) else None,
            "layer_weights": layer_weights,
            "hyperparams": hyperparams
        }
        
    def visualize_network_flow(self, sample_input, sample_label=None):
        """
        Create a visualization showing how information flows through the network.
        Works with MLP models but designed to be extensible.
        
        Args:
            sample_input: Input tensor or array
            sample_label: Optional ground truth label
            
        Returns:
            Matplotlib figure showing network flow
        """
        # Get activations through the network
        activation_info = self.trace_sample(sample_input, sample_label)
        
        # Extract model structure information
        hyperparams = activation_info["hyperparams"]
        layer_weights = activation_info["layer_weights"]
        activations = activation_info["activations"]
        
        # For MLPs, create a flow-based visualization
        if "MLP" in hyperparams["Model Type"]:
            return self._create_mlp_flow_visualization(activation_info)
        else:
            # Create a generic visualization for other model types
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Flow visualization not implemented for {hyperparams['Model Type']}", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
            
    def _create_mlp_flow_visualization(self, activation_info):
        """
        Create detailed flow visualization for MLP models with larger nodes and text
        """
        # Extract info from the activation_info
        hyperparams = activation_info["hyperparams"]
        layer_weights = activation_info["layer_weights"]
        activations = activation_info["activations"]
        sample_input = activation_info["input"]
        predicted_class = activation_info["predicted_class"]
        probabilities = activation_info["probabilities"]
        
        # Extract network dimensions
        input_size = hyperparams["Input Size"]
        hidden_sizes = hyperparams["Hidden Layers"]
        output_size = hyperparams["Output Size"]
        
        # Create figure - make it larger for better visibility
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Calculate layout dimensions
        all_layer_sizes = [input_size] + hidden_sizes + [output_size]
        max_layer_size = max(all_layer_sizes)
        layer_spacing = 4.0  # Increase horizontal spacing between layers
        
        # Create a NetworkX graph for visualization
        G = nx.DiGraph()
        
        # Add nodes for each neuron
        node_positions = {}
        neuron_colors = {}
        layer_x = 0
        
        # Add input layer
        for i in range(input_size):
            node_id = f"input_{i}"
            G.add_node(node_id)
            # Center the layer vertically
            vertical_position = (max_layer_size - input_size) / 2 + i
            node_positions[node_id] = (layer_x, vertical_position)
            
            # Color based on input value
            if torch.is_tensor(sample_input):
                input_val = sample_input[i].item() if i < len(sample_input) else 0
            else:
                input_val = sample_input[i] if i < len(sample_input) else 0
                
            # Normalize input for coloring
            intensity = min(1.0, max(0.3, abs(float(input_val)) / 5))  # Clamp between 0.3 and 1.0
            neuron_colors[node_id] = (0.4, 0.4, 0.8, intensity)
        
        # Add hidden layers
        prev_layer_ids = [f"input_{i}" for i in range(input_size)]
        layer_ids = []
        
        for l, size in enumerate(hidden_sizes):
            layer_x += layer_spacing
            layer_ids = []
            
            for i in range(size):
                node_id = f"hidden_{l}_{i}"
                G.add_node(node_id)
                layer_ids.append(node_id)
                
                # Center the layer vertically
                vertical_position = (max_layer_size - size) / 2 + i
                node_positions[node_id] = (layer_x, vertical_position)
                
                # Color based on activation value if available
                activation_key = f"linear_{l}"
                if activation_key in activations:
                    act_tensor = activations[activation_key]
                    act_val = act_tensor[0, i].item() if i < act_tensor.shape[1] else 0
                    
                    # Normalize activation for coloring
                    intensity = min(1.0, max(0.3, abs(float(act_val)) / 5))  # Clamp between 0.3 and 1.0
                    neuron_colors[node_id] = (0.3, 0.7, 0.3, intensity)
                else:
                    neuron_colors[node_id] = (0.3, 0.7, 0.3, 0.4)  # Default hidden layer color
                
                # Connect to previous layer with weighted edges
                for j, prev_id in enumerate(prev_layer_ids):
                    weight_key = f"linear_{l}" if l == 0 else f"linear_{l}"
                    if weight_key in layer_weights:
                        weight = layer_weights[weight_key]["weight"][i, j].item() if j < layer_weights[weight_key]["weight"].shape[1] else 0
                    else:
                        weight = 0
                    
                    # Add edge with weight
                    G.add_edge(prev_id, node_id, weight=weight)
            
            # Update previous layer for next connections
            prev_layer_ids = layer_ids
        
        # Add output layer
        layer_x += layer_spacing
        output_ids = []
        
        for i in range(output_size):
            node_id = f"output_{i}"
            G.add_node(node_id)
            output_ids.append(node_id)
            
            # Center the layer vertically
            vertical_position = (max_layer_size - output_size) / 2 + i
            node_positions[node_id] = (layer_x, vertical_position)
            
            # Color based on output probability
            prob = probabilities[i]
            intensity = min(1.0, max(0.3, float(prob)))  # Scale based on probability
            
            # Highlight predicted class
            if i == predicted_class:
                # Bright red for predicted class
                neuron_colors[node_id] = (0.9, 0.2, 0.2, intensity)
            else:
                # Normal output color with probability-based intensity
                neuron_colors[node_id] = (0.8, 0.3, 0.3, intensity)
            
            # Connect to previous layer with weighted edges
            for j, prev_id in enumerate(prev_layer_ids):
                if "linear_output" in layer_weights:
                    weight = layer_weights["linear_output"]["weight"][i, j].item() if j < layer_weights["linear_output"]["weight"].shape[1] else 0
                else:
                    weight = 0
                
                # Add edge with weight
                G.add_edge(prev_id, node_id, weight=weight)
        
        # Visualization parameters - increase node size
        node_sizes = {}
        for node in G.nodes():
            if "input" in node:
                node_sizes[node] = 450  # Larger input nodes
            elif "hidden" in node:
                node_sizes[node] = 450  # Larger hidden nodes
            else:  # output
                node_sizes[node] = 500  # Even larger output nodes
        
        # Edge colors based on weight polarity and magnitude
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0)
            # Normalize weight for visualization
            width = 1.5 + 2.5 * min(1, abs(weight))  # Thicker lines
            
            if weight > 0:
                edge_colors.append((0, 0.5, 0, min(0.9, 0.2 + abs(weight) * 0.7)))  # Green for positive
            else:
                edge_colors.append((0.8, 0, 0, min(0.9, 0.2 + abs(weight) * 0.7)))  # Red for negative
            
            edge_widths.append(width)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, 
            pos=node_positions, 
            node_color=[neuron_colors[node] for node in G.nodes()],
            node_size=[node_sizes[node] for node in G.nodes()],
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, 
            pos=node_positions, 
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            ax=ax,
            arrows=True,
            arrowsize=15,  # Larger arrows
            arrowstyle='->'
        )
        
        # Add labels
        # Input layer
        for i in range(input_size):
            node_id = f"input_{i}"
            x, y = node_positions[node_id]
            label = self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}"
            ax.text(x - 0.5, y, label, ha='right', va='center', fontsize=10)  # Larger font
            
            # Add input value if provided
            if torch.is_tensor(sample_input):
                val = sample_input[i].item() if i < len(sample_input) else 0
            else:
                val = sample_input[i] if i < len(sample_input) else 0
            ax.text(x + 0.05, y - 0.25, f"{val:.2f}", ha='center', va='center', fontsize=9, color='blue')  # Larger font
        
        # Output layer
        for i in range(output_size):
            node_id = f"output_{i}"
            x, y = node_positions[node_id]
            label = self.label_names[i] if i < len(self.label_names) else f"Class {i}"
            ax.text(x + 0.5, y, label, ha='left', va='center', fontsize=11)  # Larger font
            
            # Add probability
            prob = probabilities[i]
            color = 'red' if i == predicted_class else 'blue'
            ax.text(x + 0.05, y - 0.25, f"{prob:.2f}", ha='center', va='center', fontsize=9, color=color)  # Larger font
        
        # Draw layer titles
        ax.text(0, max_layer_size + 0.7, "Input Layer", ha='center', va='center', fontsize=13, fontweight='bold')
        
        for l, size in enumerate(hidden_sizes):
            layer_x = (l + 1) * layer_spacing
            ax.text(layer_x, max_layer_size + 0.7, f"Hidden Layer {l+1}", ha='center', va='center', fontsize=13, fontweight='bold')
        
        ax.text((len(hidden_sizes) + 1) * layer_spacing, max_layer_size + 0.7, "Output Layer", ha='center', va='center', fontsize=13, fontweight='bold')
        
        # Title and layout
        true_label = activation_info["true_label"]
        pred_label = activation_info["predicted_label"]
        if true_label is not None:
            ax.set_title(f"Neural Network Activation - True: {true_label}, Predicted: {pred_label}", fontsize=14)
        else:
            ax.set_title(f"Neural Network Activation - Predicted: {pred_label}", fontsize=14)
        
        ax.set_xlim(-1, (len(hidden_sizes) + 1) * layer_spacing + 1)
        ax.set_ylim(-1, max_layer_size + 1.2)  # Give more space at the top for titles
        ax.axis('off')
        
        return fig
    
    def create_interactive_activation_viz(self, sample_input, sample_label=None):
        """
        Create an interactive Plotly visualization of activations through the network
        
        Args:
            sample_input: Input tensor or array
            sample_label: Optional ground truth label
            
        Returns:
            Plotly figure with interactive activation visualization
        """
        # Get activations through the network
        activation_info = self.trace_sample(sample_input, sample_label)
        
        # Extract model structure information
        hyperparams = activation_info["hyperparams"]
        activations = activation_info["activations"]
        
        # For MLPs, create the activation visualization
        if "MLP" in hyperparams["Model Type"]:
            # Extract network dimensions
            input_size = hyperparams["Input Size"]
            hidden_sizes = hyperparams["Hidden Layers"]
            output_size = hyperparams["Output Size"]
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add activations for each layer
            layer_names = ["Input"] + [f"Hidden {i+1}" for i in range(len(hidden_sizes))] + ["Output"]
            layer_sizes = [input_size] + hidden_sizes + [output_size]
            
            # Get input values and output probabilities
            input_values = activation_info["input"]
            output_probs = activation_info["probabilities"]
            
            # Normalize inputs for visualization
            if torch.is_tensor(input_values):
                input_values = input_values.numpy()
            
            # Add input layer activations
            fig.add_trace(go.Bar(
                x=[self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}" for i in range(input_size)],
                y=input_values,
                name="Input Values",
                marker=dict(
                    color=['rgba(65, 105, 225, 0.8)'] * input_size,
                    line=dict(width=1, color='rgb(25, 25, 112)')
                ),
                hovertemplate='%{x}: %{y:.4f}<extra></extra>',
                textposition='auto',
                textfont=dict(size=14)
            ))
            
            # Add hidden layer activations
            for i, size in enumerate(hidden_sizes):
                act_key = f"linear_{i}"
                if act_key in activations:
                    act_values = activations[act_key][0].detach().numpy()
                    fig.add_trace(go.Bar(
                        x=[f"Neuron {j+1}" for j in range(size)],
                        y=act_values,
                        name=f"Hidden Layer {i+1} Activations",
                        marker=dict(
                            color=['rgba(50, 205, 50, 0.8)'] * size,
                            line=dict(width=1, color='rgb(0, 100, 0)')
                        ),
                        visible=False,
                        hovertemplate='%{x}: %{y:.4f}<extra></extra>',
                        textposition='auto',
                        textfont=dict(size=14)
                    ))
            
            # Add output layer probabilities
            fig.add_trace(go.Bar(
                x=[self.label_names[i] if i < len(self.label_names) else f"Class {i}" for i in range(output_size)],
                y=output_probs,
                name="Output Probabilities",
                marker=dict(
                    color=['rgba(220, 20, 60, 0.9)' if i == activation_info["predicted_class"] else 'rgba(128, 0, 0, 0.6)' 
                          for i in range(output_size)],
                    line=dict(width=1, color='rgb(139, 0, 0)')
                ),
                visible=False,
                hovertemplate='%{x}: %{y:.4f}<extra></extra>',
                textposition='auto',
                textfont=dict(size=14)
            ))
            
            # Create slider for navigating through layers
            steps = []
            for i, layer_name in enumerate(layer_names):
                step = {
                    'method': 'update',
                    'args': [{'visible': [j == i for j in range(len(layer_names))]},
                            {'title': f"Layer Activations: {layer_name}"}],
                    'label': layer_name
                }
                steps.append(step)
            
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Viewing: ", "font": {"size": 16}},
                pad={"t": 50},
                steps=steps
            )]
            
            # Get true and predicted labels for the title
            true_label = activation_info["true_label"] if "true_label" in activation_info else None
            pred_label = activation_info["predicted_label"]
            
            # Update layout
            title_text = f"Neural Network Activation - True: {true_label}, Predicted: {pred_label}" if true_label else f"Neural Network Activation - Predicted: {pred_label}"
            
            fig.update_layout(
                title={
                    'text': title_text,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18}
                },
                xaxis_title={"text": "Neurons", "font": {"size": 16}},
                yaxis_title={"text": "Activation Value", "font": {"size": 16}},
                sliders=sliders,
                height=600,  # Increased height
                width=1000,  # Increased width
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial"
                ),
                margin=dict(t=100, b=100, l=50, r=50)
            )
            
            # Make text on axis labels larger
            fig.update_xaxes(tickfont=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=14))
            
            return fig
        else:
            # Create placeholder for other model types
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"Activation visualization not implemented for {hyperparams['Model Type']}",
                showarrow=False,
                font=dict(size=16)
            )
            return fig 