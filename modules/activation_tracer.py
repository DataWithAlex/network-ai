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
        Create detailed flow visualization for MLP models
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate layout dimensions
        all_layer_sizes = [input_size] + hidden_sizes + [output_size]
        max_layer_size = max(all_layer_sizes)
        layer_spacing = 3.5  # Horizontal spacing between layers
        
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
            intensity = min(1.0, max(0.2, abs(float(input_val)) / 5))  # Clamp between 0.2 and 1.0
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
                    intensity = min(1.0, max(0.2, abs(float(act_val)) / 5))  # Clamp between 0.2 and 1.0
                    neuron_colors[node_id] = (0.3, 0.7, 0.3, intensity)
                else:
                    neuron_colors[node_id] = (0.3, 0.7, 0.3, 0.3)  # Default hidden layer color
                
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
            intensity = min(1.0, max(0.2, float(prob)))  # Scale based on probability
            
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
        
        # Visualization parameters
        node_sizes = {}
        for node in G.nodes():
            node_sizes[node] = 300  # Consistent node size
        
        # Edge colors based on weight polarity and magnitude
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0)
            # Normalize weight for visualization
            width = 1 + 2 * min(1, abs(weight))
            
            if weight > 0:
                edge_colors.append((0, 0.5, 0, min(0.8, 0.2 + abs(weight) * 0.6)))  # Green for positive
            else:
                edge_colors.append((0.8, 0, 0, min(0.8, 0.2 + abs(weight) * 0.6)))  # Red for negative
            
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
            alpha=0.6,
            ax=ax,
            arrows=True,
            arrowsize=10,
            arrowstyle='->'
        )
        
        # Add labels
        # Input layer
        for i in range(input_size):
            node_id = f"input_{i}"
            x, y = node_positions[node_id]
            label = self.feature_names[i] if i < len(self.feature_names) else f"Feature {i}"
            ax.text(x - 0.5, y, label, ha='right', va='center', fontsize=8)
            
            # Add input value if provided
            if torch.is_tensor(sample_input):
                val = sample_input[i].item() if i < len(sample_input) else 0
            else:
                val = sample_input[i] if i < len(sample_input) else 0
            ax.text(x + 0.05, y - 0.2, f"{val:.2f}", ha='center', va='center', fontsize=7, color='blue')
        
        # Output layer
        for i in range(output_size):
            node_id = f"output_{i}"
            x, y = node_positions[node_id]
            label = self.label_names[i] if i < len(self.label_names) else f"Class {i}"
            ax.text(x + 0.5, y, label, ha='left', va='center', fontsize=8)
            
            # Add probability
            prob = probabilities[i]
            color = 'red' if i == predicted_class else 'blue'
            ax.text(x + 0.05, y - 0.2, f"{prob:.2f}", ha='center', va='center', fontsize=7, color=color)
        
        # Draw layer titles
        ax.text(0, max_layer_size + 0.5, "Input Layer", ha='center', va='center', fontsize=10, fontweight='bold')
        
        for l, size in enumerate(hidden_sizes):
            layer_x = (l + 1) * layer_spacing
            ax.text(layer_x, max_layer_size + 0.5, f"Hidden Layer {l+1}", ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.text((len(hidden_sizes) + 1) * layer_spacing, max_layer_size + 0.5, "Output Layer", ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Title and layout
        true_label = activation_info["true_label"]
        pred_label = activation_info["predicted_label"]
        if true_label is not None:
            ax.set_title(f"Neural Network Activation - True: {true_label}, Predicted: {pred_label}")
        else:
            ax.set_title(f"Neural Network Activation - Predicted: {pred_label}")
        
        ax.set_xlim(-1, (len(hidden_sizes) + 1) * layer_spacing + 1)
        ax.set_ylim(-1, max_layer_size + 1)
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
                marker_color=['rgba(65, 105, 225, 0.7)'] * input_size
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
                        marker_color=['rgba(50, 205, 50, 0.7)'] * size,
                        visible=False
                    ))
            
            # Add output layer probabilities
            fig.add_trace(go.Bar(
                x=[self.label_names[i] if i < len(self.label_names) else f"Class {i}" for i in range(output_size)],
                y=output_probs,
                name="Output Probabilities",
                marker_color=['rgba(220, 20, 60, 0.7)' if i == activation_info["predicted_class"] else 'rgba(128, 0, 0, 0.5)' 
                              for i in range(output_size)],
                visible=False
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
                steps=steps
            )]
            
            # Update layout
            fig.update_layout(
                title=f"Activation Values Through Network Layers",
                xaxis_title="Neurons",
                yaxis_title="Activation Value",
                sliders=sliders,
                height=500,
                width=900
            )
            
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
     # ---------------------------------------------------------------------
    #  DROP-IN REPLACEMENT  (paste over the existing method)
    # ---------------------------------------------------------------------
    def create_interactive_network_flow(self, sample_input, sample_label=None):
        """
        Interactive Plotly visualisation of a forward pass.
        * Feature names shown left of input nodes.
        * Class names shown right of output nodes.
        * Hidden-layer numbers are post-activation values.
        * Edges stop at node rims; hover reveals weight value.
        * Generalises to any number of hidden layers / any dataset.
        """
        import math
        import plotly.graph_objects as go

        # ------------------------------------------------ visual config --
        visual_config = {
            # Node appearance
            "node_radius": 0.35,           # Node "radius" in data units (for edge connections)
            "node_size": 40,               # Size of nodes in pixels
            "node_border_width": 2,        # Width of node border
            "node_border_color": "black",  # Color of node border
            
            # Layout spacing
            "x_gap": 25,                   # Horizontal spacing between layers
            "y_gap": 8,                    # Vertical spacing between nodes in same layer
            
            # Text appearance
            "font_size": 12,               # Font size for node text and labels
            "input_label_offset": -8,      # X-offset for input feature labels
            "output_label_offset": 8,      # X-offset for output class labels
            
            # Edge appearance
            "edge_min_width": 0.5,         # Minimum edge width
            "edge_max_width": 5,           # Maximum edge width
            "edge_width_scale": 3,         # Scaling factor for edge width
            "edge_min_opacity": 0.3,       # Minimum edge opacity
            "edge_max_opacity": 0.9,       # Maximum edge opacity
            "edge_weight_threshold": 0.03, # Minimum weight to show an edge
            
            # Colors
            "input_node_color": "rgba(65,105,225,0.9)",   # Blue
            "hidden_node_color": "rgba(50,205,50,0.9)",   # Green
            "output_node_color": "rgba(220,20,60,0.9)",   # Red
            "predicted_node_color": "rgba(255,215,0,0.9)", # Gold
            "positive_edge_color": "rgba(0,128,0,{alpha})", # Green
            "negative_edge_color": "rgba(255,0,0,{alpha})", # Red
            
            # Figure dimensions
            "figure_width": 1400,          # Width of figure in pixels
            "figure_height": 800,          # Height of figure in pixels
            "margin_left": 40,             # Left margin
            "margin_right": 40,            # Right margin
            "margin_top": 100,              # Top margin
            "margin_bottom": 40,           # Bottom margin
            "plot_bg_color": "white"       # Background color
        }

        model, feat, lab = self.model, self.feature_names, self.label_names

        # --------------------------------------------------- forward pass --
        acts = {}
        def make_hook(name):
            def _hook(_, __, out): acts[name] = out.detach()
            return _hook
        hooks = []
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.ReLU)):
                hooks.append(m.register_forward_hook(make_hook(n)))
        with torch.no_grad():
            logits = model(sample_input.unsqueeze(0))
            probs  = torch.softmax(logits, 1)[0]
            pred   = torch.argmax(probs).item()
        for h in hooks:
            h.remove()

        # ------------------------------------------------ network anatomy --
        hp            = model.get_hyperparameters()
        hidden_sizes  = hp['Hidden Layers']
        n_in          = hp['Input Size']
        n_out         = hp['Output Size']
        sizes         = [n_in] + hidden_sizes + [n_out]

        layer_weights = model.get_layer_weights()
        w_keys        = [f'linear_{i}' for i in range(len(hidden_sizes))] + ['linear_output']
        weights       = [layer_weights[k]['weight'] for k in w_keys]

        # ------------------------------------------------ layout variables --
        node_R          = visual_config["node_radius"]
        node_px         = visual_config["node_size"]
        x_gap           = visual_config["x_gap"]
        y_gap           = visual_config["y_gap"]
        max_nodes       = max(sizes)
        y0              = (max_nodes - 1) * y_gap / 2     # centre vertically
        font_sz         = visual_config["font_size"]

        node_xy = {}
        traces  = []

        def add_node(node_id, x, y, txt, col):
            node_xy[node_id] = (x, y)
            traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=node_px, 
                    color=col, 
                    line=dict(
                        width=visual_config["node_border_width"],
                        color=visual_config["node_border_color"]
                    )
                ),
                text=[txt], textposition="middle center",
                textfont=dict(size=font_sz, color='white'),
                hoverinfo='text', hovertext=node_id,
                showlegend=False))

        # ------------------------------------------------ build the graph --
        # input layer
        for i in range(n_in):
            y = i * y_gap - y0
            val = float(sample_input[i])
            add_node(f'in_{i}', 0, y, f'{val:.2f}', visual_config["input_node_color"])
            # feature name annotation
            fname = feat[i] if i < len(feat) else f'Feature {i}'
            traces.append(go.Scatter(
                x=[visual_config["input_label_offset"]], y=[y],
                mode='text', text=[fname],
                textfont=dict(size=font_sz, color='black'),
                hoverinfo='skip', showlegend=False))

        # hidden layers
        for li, h_size in enumerate(hidden_sizes):
            x = (li + 1) * x_gap
            post_act = acts.get(f'activation_{li}', acts.get(f'linear_{li}'))
            for j in range(h_size):
                y = j * y_gap - y0
                val = float(post_act[0, j]) if post_act is not None else 0.0
                add_node(f'h{li}_{j}', x, y, f'{val:.2f}', visual_config["hidden_node_color"])

        # output layer
        x_out = (len(hidden_sizes) + 1) * x_gap
        for i in range(n_out):
            y = i * y_gap - y0
            col = visual_config["predicted_node_color"] if i == pred else visual_config["output_node_color"]
            add_node(f'out_{i}', x_out, y, f'{float(probs[i]):.2f}', col)
            cname = lab[i] if i < len(lab) else f'Class {i}'
            traces.append(go.Scatter(
                x=[x_out + visual_config["output_label_offset"]], y=[y],
                mode='text', text=[cname],
                textfont=dict(size=font_sz, color='black'),
                hoverinfo='skip', showlegend=False))

        # -------------------------------------------- helper: draw an edge --
        def draw_edge(n0, n1, w):
            x0,y0 = node_xy[n0]; x1,y1 = node_xy[n1]
            d     = math.hypot(x1-x0, y1-y0)
            if d == 0: return
            sx, sy = x0 + (x1-x0)*node_R/d, y0 + (y1-y0)*node_R/d
            ex, ey = x1 - (x1-x0)*node_R/d, y1 - (y1-y0)*node_R/d
            alpha  = visual_config["edge_min_opacity"] + (visual_config["edge_max_opacity"] - visual_config["edge_min_opacity"])*min(1, abs(w))
            col    = visual_config["positive_edge_color"].format(alpha=alpha) if w>0 else visual_config["negative_edge_color"].format(alpha=alpha)
            width  = min(visual_config["edge_max_width"], visual_config["edge_min_width"] + visual_config["edge_width_scale"]*abs(w))
            traces.append(go.Scatter(
                x=[sx,ex], y=[sy,ey], mode='lines',
                line=dict(color=col, width=width),
                hoverinfo='text',
                hovertext=f'{n0} → {n1}<br>weight: {w:.4f}',
                showlegend=False))

        # --------------------------------------------- connect every layer --
        layer_ids = (
            [[f'in_{i}'        for i in range(n_in)]] +
            [[f'h{li}_{j}'     for j in range(sz)] for li, sz in enumerate(hidden_sizes)] +
            [[f'out_{i}'       for i in range(n_out)]]
        )

        for L, W in enumerate(weights):
            from_ids, to_ids = layer_ids[L], layer_ids[L+1]
            for j, to_id in enumerate(to_ids):
                for i, from_id in enumerate(from_ids):
                    if i < W.shape[1] and j < W.shape[0]:
                        w = float(W[j, i])
                        if abs(w) > visual_config["edge_weight_threshold"]:
                            draw_edge(from_id, to_id, w)

        # ---------------------------------------------------- final layout --
        true_lab = lab[sample_label.item()] if sample_label is not None else '?'
        fig = go.Figure(traces)
        fig.update_layout(
            title=f"Neural Network Activation – True: {true_lab}, Predicted: {lab[pred]}",
            xaxis=dict(visible=False, range=[-2, x_out+2]),
            yaxis=dict(visible=False, range=[-y0-2, y0+2], scaleanchor='x', scaleratio=1),
            width=visual_config["figure_width"], 
            height=visual_config["figure_height"], 
            plot_bgcolor=visual_config["plot_bg_color"],
            hovermode='closest', 
            margin=dict(
                l=visual_config["margin_left"],
                r=visual_config["margin_right"],
                t=visual_config["margin_top"],
                b=visual_config["margin_bottom"]
            )
        )
        return fig