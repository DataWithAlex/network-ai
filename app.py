import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import our modules
from modules.data_loader import get_dataset
from modules.model_builder import create_model
from modules.trainer import Trainer
from modules.visualizer import NetworkVisualizer

# Set page configuration
st.set_page_config(page_title="Neural Network Visualizer", page_icon="🧠", layout="wide")

# Title and description
st.title("Neural Network Visualizer")
st.markdown("""
This application helps visualize neural networks through the lens of graph theory to address the "black box" problem. 
Explore how data flows through the network and how it transforms at each layer.
""")

# Sidebar for dataset selection and model parameters
st.sidebar.header("Settings")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Titanic"]
)

# Model type selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["MLP"]
)

# Model parameters
st.sidebar.subheader("Model Architecture")
if dataset_name == "Iris":
    default_hidden_sizes = [8, 4]
else:
    default_hidden_sizes = [16, 8]

hidden_layer_1 = st.sidebar.slider("Neurons in Hidden Layer 1", 2, 32, default_hidden_sizes[0])
hidden_layer_2 = st.sidebar.slider("Neurons in Hidden Layer 2", 2, 32, default_hidden_sizes[1])
hidden_sizes = [hidden_layer_1, hidden_layer_2]

dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.0, 0.1)

# Training parameters
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    value=0.001
)
epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Initialize session state (to store trained model)
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = None
if 'sample_datapoint' not in st.session_state:
    st.session_state.sample_datapoint = None
if 'sample_label' not in st.session_state:
    st.session_state.sample_label = None

# Function to load data and train model
def load_and_train():
    # Load and preprocess dataset
    dataset = get_dataset(dataset_name, batch_size=batch_size)
    data_info = dataset.load_data()
    
    # Create model
    model = create_model(
        model_type=model_type,
        input_size=data_info["n_features"],
        hidden_sizes=hidden_sizes,
        output_size=data_info["n_classes"],
        dropout_rate=dropout_rate
    )
    
    # Create trainer
    trainer = Trainer(model, learning_rate=learning_rate)
    
    # Train model
    with st.spinner("Training model..."):
        training_history = trainer.train(
            data_info["train_loader"], 
            data_info["test_loader"], 
            epochs=epochs, 
            verbose=False
        )
    
    # Evaluate model
    performance_metrics = trainer.get_performance_metrics(
        data_info["test_loader"], 
        data_info["label_names"]
    )
    
    # Create visualizer
    visualizer = NetworkVisualizer(model, data_info["feature_names"], data_info["label_names"])
    
    # Sample a random datapoint for activation visualization
    sample_idx = np.random.randint(0, len(data_info["test_dataset"]))
    sample_datapoint, sample_label = data_info["test_dataset"][sample_idx]
    
    # Store in session state
    st.session_state.model = model
    st.session_state.trainer = trainer
    st.session_state.data_info = data_info
    st.session_state.visualizer = visualizer
    st.session_state.training_history = training_history
    st.session_state.performance_metrics = performance_metrics
    st.session_state.sample_datapoint = sample_datapoint
    st.session_state.sample_label = sample_label
    
    return {
        'model': model,
        'trainer': trainer,
        'data_info': data_info,
        'visualizer': visualizer,
        'training_history': training_history,
        'performance_metrics': performance_metrics,
        'sample_datapoint': sample_datapoint,
        'sample_label': sample_label
    }

# Button to train model
if st.sidebar.button("Train Model"):
    load_and_train()
    st.sidebar.success("Model trained successfully!")

# Tabs for different visualizations
if st.session_state.model is not None:
    # Add a new tab for model information
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Model Information",
        "Network Architecture", 
        "Training History", 
        "Feature Importance", 
        "Data Projections",
        "Activation Visualization"
    ])
    
    # Tab 1: Model Information
    with tab1:
        st.header("Model Information")
        
        # Display dataset information
        st.subheader("Dataset Information")
        dataset = get_dataset(dataset_name)
        st.markdown(dataset.get_description())
        
        st.markdown(f"""
        **Dataset Statistics:**
        - Number of features: {st.session_state.data_info['n_features']}
        - Number of classes: {st.session_state.data_info['n_classes']}
        - Feature names: {', '.join(st.session_state.data_info['feature_names'])}
        - Class names: {', '.join(st.session_state.data_info['label_names'])}
        """)
        
        # Display model hyperparameters
        st.subheader("Model Hyperparameters")
        hyperparams = st.session_state.model.get_hyperparameters()
        
        # Convert hyperparameters to DataFrame for display
        hyperparams_df = pd.DataFrame({
            'Parameter': hyperparams.keys(),
            'Value': hyperparams.values()
        })
        
        st.dataframe(hyperparams_df)
        
        # Display model description
        st.subheader("Model Description")
        st.markdown(st.session_state.model.get_description())
    
    # Tab 2: Network Architecture
    with tab2:
        st.header("Neural Network Architecture")
        st.markdown("""
        This interactive visualization shows the complete architecture of the neural network:
        
        ### Layer Types:
        - **Blue blocks**: Input layer nodes representing dataset features
        - **Green blocks**: Fully connected (dense) layers with weights and biases
        - **Purple circles**: Activation functions that introduce non-linearity
        - **Gray blocks**: Dropout layers (regularization technique)
        - **Red blocks**: Output layer representing classes
        
        ### Visualization Features:
        - **Hover over any component** to see detailed statistics including weight distributions
        - **Connections between layers** show data flow and transformation
        - **Dotted lines** provide explanatory annotations
        - **Connection color** indicates weight polarity (green=positive, red=negative)
        - **Connection thickness** indicates weight magnitude
        
        This visualization helps understand how data flows through the network and how each layer transforms the data.
        """)
        
        # Display the interactive visualization
        st.subheader("Interactive Architecture Diagram")
        fig = st.session_state.visualizer.plot_network_interactive()
        st.plotly_chart(fig, use_container_width=True)
        
        # Model parameter details
        st.subheader("Model Parameter Details")
        hyperparams = st.session_state.model.get_hyperparameters()
        st.write(f"**Total Parameters:** {hyperparams['Total Parameters']:,}")
        
        # Display parameter breakdown by layer
        weights = st.session_state.model.get_layer_weights()
        
        # Create a dataframe of layer parameters
        layer_params = []
        total_params = 0
        
        for layer_name, weight_data in weights.items():
            weight_count = np.prod(weight_data['weight'].shape)
            bias_count = len(weight_data['bias']) if weight_data['bias'] is not None else 0
            params = weight_count + bias_count
            total_params += params
            
            layer_params.append({
                'Layer': layer_name,
                'Weight Shape': str(weight_data['weight'].shape),
                'Parameters': params,
                'Percentage': f"{100 * params / hyperparams['Total Parameters']:.1f}%"
            })
        
        param_df = pd.DataFrame(layer_params)
        st.dataframe(param_df)
        
        # Keep the static visualization as an option
        if st.checkbox("Show Traditional Network Diagram"):
            st.subheader("Traditional Network Diagram")
            fig = st.session_state.visualizer.plot_network_architecture()
            st.pyplot(fig)
        
        # Add sample input tracing
        st.subheader("Sample Input Tracing")
        st.markdown("""
        This visualization shows how a sample input propagates through the network. Select a sample to see how the model
        processes it to make a prediction. Node colors show activation strength, and edge colors/thickness show weight importance.
        """)
        
        # Sample selection
        if st.button("Select Random Sample"):
            data_info = st.session_state.data_info
            test_dataset = data_info["test_dataset"]
            sample_idx = np.random.randint(0, len(test_dataset))
            st.session_state.sample_datapoint, st.session_state.sample_label = test_dataset[sample_idx]
        
        # Display the sample details
        if hasattr(st.session_state, 'sample_datapoint') and st.session_state.sample_datapoint is not None:
            sample_datapoint = st.session_state.sample_datapoint
            sample_label = st.session_state.sample_label
            
            # Create two columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display sample features
                st.subheader("Sample Features")
                
                # Feature names and values - ensure all values are properly converted to Python native types
                feature_names = st.session_state.data_info['feature_names']
                
                # Safely convert tensor values to native Python float values
                if torch.is_tensor(sample_datapoint):
                    feature_values = [float(val.item()) for val in sample_datapoint]
                else:
                    feature_values = [float(val) for val in sample_datapoint]
                    
                sample_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': feature_values
                })
                
                # Display the features table
                st.dataframe(sample_df)
                
                # Show true label
                true_class = st.session_state.data_info['label_names'][sample_label.item()]
                st.markdown(f"**True Class**: {true_class}")
                
                # Make prediction
                with torch.no_grad():
                    output = st.session_state.model(sample_datapoint.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probs).item()
                    predicted_class = st.session_state.data_info['label_names'][prediction]
                
                # Show prediction
                st.markdown(f"**Predicted Class**: {predicted_class} " + 
                           ("✓" if prediction == sample_label.item() else "✗"))
                
                # Show top probabilities
                prob_df = pd.DataFrame({
                    'Class': st.session_state.data_info['label_names'],
                    'Probability': probs.numpy()
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(prob_df)
            
            with col2:
                # Display network with activations
                st.subheader("Activation Path")
                fig = st.session_state.visualizer.plot_detailed_network(
                    sample_input=sample_datapoint,
                    sample_label=sample_label
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a sample to see how it propagates through the network.")
    
    # Tab 3: Training History
    with tab3:
        st.header("Training History")
        st.markdown("""
        These plots show the model's performance during training.
        - Loss: The objective function being minimized
        - Accuracy: The percentage of correct predictions
        """)
        
        # Plot training history
        history = st.session_state.training_history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss During Training')
        axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy During Training')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display performance metrics
        st.subheader("Model Performance Metrics")
        
        metrics = st.session_state.performance_metrics
        st.metric("Test Accuracy", f"{metrics['accuracy']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(metrics['confusion_matrix'], cmap='Blues')
            
            # Add labels
            ax.set_xticks(np.arange(len(st.session_state.data_info['label_names'])))
            ax.set_yticks(np.arange(len(st.session_state.data_info['label_names'])))
            ax.set_xticklabels(st.session_state.data_info['label_names'])
            ax.set_yticklabels(st.session_state.data_info['label_names'])
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            plt.colorbar(im)
            
            # Add text annotations
            for i in range(len(st.session_state.data_info['label_names'])):
                for j in range(len(st.session_state.data_info['label_names'])):
                    text = ax.text(j, i, metrics['confusion_matrix'][i, j],
                                  ha="center", va="center", color="white" if metrics['confusion_matrix'][i, j] > metrics['confusion_matrix'].max()/2 else "black")
            
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            fig.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.subheader("Classification Report")
            st.text(metrics['classification_report'])
    
    # Tab 4: Feature Importance
    with tab4:
        st.header("Feature Importance")
        st.markdown("""
        This visualization shows the importance of each input feature based on the weights in the first layer of the network.
        Larger absolute weight values indicate higher importance of that feature.
        """)
        
        fig = st.session_state.visualizer.visualize_feature_importance()
        st.pyplot(fig)
    
    # Tab 5: Data Projections
    with tab5:
        st.header("Data Projections")
        st.markdown("""
        These visualizations show how the data is transformed as it passes through the network.
        - PCA (Principal Component Analysis) reduces the dimensionality of the data to 2 dimensions, preserving as much variance as possible.
        - t-SNE (t-distributed Stochastic Neighbor Embedding) reduces the dimensionality while trying to preserve the local structure of the data.
        
        Colors represent different classes.
        """)
        
        # Get a batch of data for visualization
        projection_data = st.session_state.data_info['original_X']
        projection_labels = st.session_state.data_info['original_y']
        
        # Get sample datapoint for layer activations
        sample_tensor = st.session_state.sample_datapoint.unsqueeze(0)  # Add batch dimension
        layer_outputs = st.session_state.model.get_intermediate_features(sample_tensor)
        
        # Sample a small subset if dataset is large
        if len(projection_labels) > 500:
            # Sample a balanced subset
            indices = []
            for label in np.unique(projection_labels):
                label_indices = np.where(projection_labels == label)[0]
                sampled_indices = np.random.choice(label_indices, size=min(100, len(label_indices)), replace=False)
                indices.extend(sampled_indices)
            
            projection_data = projection_data[indices]
            projection_labels = projection_labels[indices]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PCA Projection")
            fig = st.session_state.visualizer.visualize_pca_projection(
                projection_data, projection_labels
            )
            st.pyplot(fig)
            
        with col2:
            st.subheader("t-SNE Projection")
            perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
            fig = st.session_state.visualizer.visualize_tsne_projection(
                projection_data, projection_labels, perplexity=perplexity
            )
            st.pyplot(fig)
        
        # Visualize layer projections
        st.subheader("Layer-wise Data Projections")
        st.markdown("""
        These visualizations show how the data is transformed at each layer of the network.
        """)
        
        # Get activations for a batch of data
        with torch.no_grad():
            batch_size = min(100, len(projection_data))
            batch_data = torch.tensor(projection_data[:batch_size], dtype=torch.float32)
            batch_labels = projection_labels[:batch_size]
            layer_outputs_batch = st.session_state.model.get_intermediate_features(batch_data)
        
        projection_type = st.radio(
            "Projection Method",
            ["PCA", "t-SNE"]
        )
        
        if projection_type == "PCA":
            fig = st.session_state.visualizer.visualize_pca_projection(
                batch_data.numpy(), batch_labels, layer_outputs_batch
            )
        else:  # t-SNE
            fig = st.session_state.visualizer.visualize_tsne_projection(
                batch_data.numpy(), batch_labels, layer_outputs_batch, perplexity=perplexity
            )
        
        st.pyplot(fig)
    
    # Tab 6: Activation Visualization
    with tab6:
        st.header("Neuron Activations")
        st.markdown("""
        This visualization shows the activation values of each neuron in the network for a specific input example.
        It helps understand how information flows through the network.
        """)
        
        # Display sample information
        sample_datapoint = st.session_state.sample_datapoint
        sample_label = st.session_state.sample_label
        label_name = st.session_state.data_info['label_names'][sample_label]
        
        st.subheader(f"Sample Data Point (Class: {label_name})")
        
        # Show feature values
        feature_values = {}
        for i, feature in enumerate(st.session_state.data_info['feature_names']):
            if i < len(sample_datapoint):
                feature_values[feature] = sample_datapoint[i].item()
        
        st.dataframe(pd.DataFrame([feature_values]))
        
        # Generate new sample button
        if st.button("Generate New Sample"):
            data_info = st.session_state.data_info
            test_dataset = data_info["test_dataset"]
            sample_idx = np.random.randint(0, len(test_dataset))
            st.session_state.sample_datapoint, st.session_state.sample_label = test_dataset[sample_idx]
            st.experimental_rerun()
        
        # Display activations
        st.subheader("Neuron Activations")
        fig = st.session_state.visualizer.visualize_activations(sample_datapoint)
        st.pyplot(fig)
        
        # Get prediction
        with torch.no_grad():
            output = st.session_state.model(sample_datapoint.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            predicted_label = st.session_state.data_info['label_names'][prediction]
        
        # Show prediction and probabilities
        st.subheader("Model Prediction")
        
        # Display prediction
        correct = prediction == sample_label.item()
        st.markdown(f"**Predicted Class**: {predicted_label} " + 
                   ("✓" if correct else "✗"))
        
        # Display probabilities
        prob_df = pd.DataFrame({
            'Class': st.session_state.data_info['label_names'],
            'Probability': probabilities.numpy()
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(prob_df['Class'], prob_df['Probability'], color='skyblue')
        
        # Highlight the correct class
        bars[sample_label.item()].set_color('green')
        
        # Highlight the predicted class if different from correct
        if prediction != sample_label.item():
            bars[prediction].set_color('red')
            
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color='green', label='True Class'),
        ]
        if prediction != sample_label.item():
            legend_elements.append(mpatches.Patch(color='red', label='Predicted Class (Wrong)'))
            
        ax.legend(handles=legend_elements)
        
        st.pyplot(fig)

else:
    st.info("Please train a model to see visualizations.")

# Footer with info
st.markdown("---")
st.markdown("""
### Understanding Neural Networks through Graph Theory

This application visualizes neural networks as graphs, where:
- **Nodes** represent neurons in the network
- **Edges** represent the connections (weights) between neurons
- **Layers** represent different stages of data transformation

By exploring these visualizations, you can gain insights into:
1. How the network architecture affects its learning capacity
2. How important each feature is for the model's predictions
3. How the data is transformed as it flows through the network
4. How activations at each layer contribute to the final prediction

This approach helps make the "black box" of neural networks more transparent.
""")

if __name__ == "__main__":
    pass