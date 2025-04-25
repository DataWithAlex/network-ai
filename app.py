import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# Import our modules
from modules.data_loader import get_dataset
from modules.model_builder import create_model
from modules.trainer import Trainer
from modules.visualizer import NetworkVisualizer
from modules.activation_tracer import ActivationTracer

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
if 'activation_tracer' not in st.session_state:
    st.session_state.activation_tracer = None
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
    activation_tracer = ActivationTracer(model, data_info["feature_names"], data_info["label_names"])
    
    
    # Sample a random datapoint for activation visualization
    sample_idx = np.random.randint(0, len(data_info["test_dataset"]))
    sample_datapoint, sample_label = data_info["test_dataset"][sample_idx]
    
    # Store in session state
    st.session_state.model = model
    st.session_state.trainer = trainer
    st.session_state.data_info = data_info
    st.session_state.visualizer = visualizer
    st.session_state.activation_tracer = activation_tracer
    st.session_state.training_history = training_history
    st.session_state.performance_metrics = performance_metrics
    st.session_state.sample_datapoint = sample_datapoint
    st.session_state.sample_label = sample_label
    
    return {
        'model': model,
        'trainer': trainer,
        'data_info': data_info,
        'visualizer': visualizer,
        'activation_tracer': activation_tracer,
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
        
        # Display dataset information with themed icons
        st.subheader("Dataset Information")
        dataset = get_dataset(dataset_name)
        
        # Add themed icons based on dataset
        col1, col2 = st.columns([1, 3])
        with col1:
            if dataset_name == "Iris":
                st.image("assets/iris.png", width=150, caption="Iris Dataset")
            elif dataset_name == "Titanic":
                st.image("assets/titanic.jpg", width=150, caption="Titanic Dataset")
        
        with col2:
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
        This interactive visualization shows the complete architecture of the neural network. 
        **Hover your mouse over any component** to see detailed information.
        
        ### Layer Types:
        - **Blue blocks**: Input layer nodes representing dataset features
        - **Green blocks**: Fully connected (dense) layers with weights and biases
        - **Purple circles**: Activation functions that introduce non-linearity
        - **Gray blocks**: Dropout layers (regularization technique)
        - **Red blocks**: Output layer representing classes
        
        ### How to Use:
        1. **Hover** over any node to see its details and activation values
        2. **Observe the connections** between layers to understand data flow
        3. **Note the color and thickness** of connections to see weight importance
        """)
        
        # Display the interactive visualization
        st.subheader("Interactive Architecture Diagram")
        st.markdown("""
        Hover over nodes to see details about each layer. 
        The colors indicate different layer types as shown in the legend.
        """)
        
        # Get and display the network visualization
        try:
            network_fig = st.session_state.visualizer.plot_network_interactive()
            st.plotly_chart(network_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
        
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
            fig = st.session_state.visualizer.plot_network_interactive()
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            st.subheader("Sample Data Propagation")
            st.markdown("See how this specific data point flows through the network and activates neurons.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced sample data display with themed visuals
                st.markdown("### Input Features")
                
                # Create a more visually appealing feature display
                feature_df = pd.DataFrame({
                    'Feature': st.session_state.data_info['feature_names'],
                    'Value': sample_datapoint.numpy()
                })
                
                # Add themed visualization based on dataset
                if dataset_name == "Iris":
                    st.markdown("""
                    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4682b4;">
                        <h4 style="color: #4682b4;">🌸 Iris Sample</h4>
                        <p>This sample represents measurements from an iris flower.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a radar chart for iris features
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=sample_datapoint.numpy(),
                        theta=st.session_state.data_info['feature_names'],
                        fill='toself',
                        name='Sample Features',
                        line_color='rgb(70, 130, 180)'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif dataset_name == "Titanic":
                    st.markdown("""
                    <div style="background-color: #f0f5ff; padding: 15px; border-radius: 10px; border-left: 5px solid #000080;">
                        <h4 style="color: #000080;">🚢 Passenger Information</h4>
                        <p>This sample represents a passenger on the Titanic.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a more informative display for Titanic data
                    categorical_features = ['Sex', 'Class', 'Embarked']
                    numerical_features = [f for f in st.session_state.data_info['feature_names'] if f not in categorical_features]
                    
                    # Display categorical features as icons/badges
                    for i, feature in enumerate(st.session_state.data_info['feature_names']):
                        value = sample_datapoint[i].item()
                        if feature in categorical_features:
                            if feature == 'Sex':
                                icon = "👨" if value > 0.5 else "👩"
                                label = "Male" if value > 0.5 else "Female"
                            elif feature == 'Class':
                                if value < 0.33:
                                    icon, label = "🥇", "1st Class"
                                elif value < 0.66:
                                    icon, label = "🥈", "2nd Class"
                                else:
                                    icon, label = "🥉", "3rd Class"
                            elif feature == 'Embarked':
                                ports = ["Cherbourg", "Queenstown", "Southampton"]
                                idx = min(int(value * 3), 2)
                                icon, label = "🚢", ports[idx]
                            
                            st.markdown(f"**{feature}**: {icon} {label}")
                
                # Display the feature table for all datasets
                st.dataframe(feature_df)
            
            with col2:
                # Show true and predicted class with enhanced visuals
                st.markdown("### Classification Results")
                
                # Get true label
                true_class = st.session_state.data_info['label_names'][sample_label.item()]
                
                # Make prediction
                with torch.no_grad():
                    output = st.session_state.model(sample_datapoint.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probs).item()
                    predicted_class = st.session_state.data_info['label_names'][prediction]
                
                # Create a more visually appealing result display
                is_correct = prediction == sample_label.item()
                result_color = "#28a745" if is_correct else "#dc3545"
                icon = "✅" if is_correct else "❌"
                
                st.markdown(f"""
                <div style="background-color: rgba({','.join(['40, 167, 69, 0.1' if is_correct else '220, 53, 69, 0.1'])});
                     padding: 15px; border-radius: 10px; border-left: 5px solid {result_color};">
                    <h4>Prediction Result {icon}</h4>
                    <p><b>True Class:</b> {true_class}</p>
                    <p><b>Predicted Class:</b> {predicted_class}</p>
                    <p><b>Confidence:</b> {probs[prediction].item()*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show class-specific visuals based on dataset
                if dataset_name == "Iris":
                    # Display the main iris image instead of individual species images
                    st.markdown("### Iris Classification")
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                        <p>Predicted class: <b>{predicted_class}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Display the iris image in full width
                    st.image("assets/iris.png", caption="Iris Species Comparison", use_column_width=True)
                elif dataset_name == "Titanic":
                    # Display the Titanic image
                    st.markdown("### Titanic Passenger Classification")
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                        <p>Predicted survival status: <b>{predicted_class}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Display the Titanic image in full width
                    st.image("assets/titanic.jpg", caption="RMS Titanic", use_column_width=True)
                
                # Show probabilities as a horizontal bar chart
                st.markdown("### Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': st.session_state.data_info['label_names'],
                    'Probability': probs.numpy()
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                            color='Probability', color_continuous_scale='Blues',
                            text=prob_df['Probability'].apply(lambda x: f"{x:.2%}"))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display network with activations using full width
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
        st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Classification Report")
            st.text(metrics['classification_report'])
    
    # Tab 4: Feature Importance
    with tab4:
        st.header("Feature Importance Analysis")
        st.markdown("""
        This visualization shows how each input feature impacts the model's predictions:
        
        - **Higher values** indicate features that have a stronger influence on the model's decisions
        - The visualization uses the weights from the first layer of the neural network
        - For more complex models, this is an approximation of feature importance
        """)

        # Get original data from session state
        if hasattr(st.session_state, 'data_info') and 'original_X' in st.session_state.data_info:
            original_X = st.session_state.data_info['original_X']
            original_y = st.session_state.data_info['original_y']
            
            # Pass the data to the visualization method
            fig = st.session_state.visualizer.visualize_feature_importance(X=original_X, y=original_y)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Original dataset information not available in session state")
    
    # Tab 5: Data Projections
    with tab5:
        st.header("Data Projections Visualization")
        st.markdown("""
        These visualizations show how the neural network transforms the data through its layers:
        
        - **PCA Projection** reduces the dimensionality to 2D while preserving variance
        - **t-SNE Projection** preserves local neighborhood structure of the data
        - Colors represent different classes in the dataset
        """)
        
        # Create tabs for PCA and t-SNE
        proj_tab1, proj_tab2 = st.tabs(["PCA Projection", "t-SNE Projection"])
        
        with proj_tab1:
            # Get original data from session state
            if hasattr(st.session_state, 'data_info') and 'original_X' in st.session_state.data_info:
                original_X = st.session_state.data_info['original_X']
                original_y = st.session_state.data_info['original_y']
                
                # Get intermediate features for all training examples (limit to 1000 for performance)
                max_samples = min(1000, len(original_X))
                subset_X = original_X[:max_samples]
                subset_y = original_y[:max_samples]
                
                # Get intermediate features if available
                with torch.no_grad():
                    # Convert to tensor if needed
                    if not torch.is_tensor(subset_X):
                        subset_X_tensor = torch.tensor(subset_X, dtype=torch.float32)
                    else:
                        subset_X_tensor = subset_X
                    
                    # Get intermediate features for visualization
                    intermediate_features = st.session_state.model.get_intermediate_features(subset_X_tensor)
                
                # Create PCA visualization
                fig = st.session_state.visualizer.visualize_pca_projection(
                    X=subset_X, 
                    y=subset_y, 
                    layer_outputs=intermediate_features
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Original dataset information not available in session state")
        
        with proj_tab2:
            # Get original data from session state
            if hasattr(st.session_state, 'data_info') and 'original_X' in st.session_state.data_info:
                original_X = st.session_state.data_info['original_X']
                original_y = st.session_state.data_info['original_y']
                
                # Get intermediate features for all training examples (limit to 500 for t-SNE which is slower)
                max_samples = min(500, len(original_X))
                subset_X = original_X[:max_samples]
                subset_y = original_y[:max_samples]
                
                # Allow user to adjust perplexity
                perplexity = st.slider("t-SNE Perplexity", min_value=5, max_value=50, value=30,
                                      help="Perplexity is related to the number of nearest neighbors in t-SNE")
                
                # Get intermediate features if available
                with torch.no_grad():
                    # Convert to tensor if needed
                    if not torch.is_tensor(subset_X):
                        subset_X_tensor = torch.tensor(subset_X, dtype=torch.float32)
                    else:
                        subset_X_tensor = subset_X
                    
                    # Get intermediate features for visualization
                    intermediate_features = st.session_state.model.get_intermediate_features(subset_X_tensor)
                
                # Create t-SNE visualization
                fig = st.session_state.visualizer.visualize_tsne_projection(
                    X=subset_X, 
                    y=subset_y, 
                    layer_outputs=intermediate_features,
                    perplexity=perplexity
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Original dataset information not available in session state")
    
    # Tab 6: Activation Visualization
    with tab6:
        st.header("Neuron Activations")
        st.markdown("""
        This visualization shows the activation values of each neuron in the network for a specific input example.
        It helps understand how information flows through the network.
        """)
        
        # Generate new sample button
        if st.button("Generate New Sample", key="activation_new_sample"):
            data_info = st.session_state.data_info
            test_dataset = data_info["test_dataset"]
            sample_idx = np.random.randint(0, len(test_dataset))
            st.session_state.sample_datapoint, st.session_state.sample_label = test_dataset[sample_idx]
            st.experimental_rerun()
        
        # Sample info and prediction
        if hasattr(st.session_state, 'sample_datapoint') and st.session_state.sample_datapoint is not None:
            sample_datapoint = st.session_state.sample_datapoint
            sample_label = st.session_state.sample_label
            
            # Create two columns for sample information
            col1, col2 = st.columns(2)
            
            with col1:
                # Display sample features
                st.subheader("Sample Features")
                
                # Feature names and values
                feature_names = st.session_state.data_info['feature_names']
                
                # Safely convert tensor values to native Python float values
                if torch.is_tensor(sample_datapoint):
                    feature_values = [float(val.item()) for val in sample_datapoint]
                else:
                    feature_values = [float(val) for val in sample_datapoint]
                    
                # Create and display dataframe
                sample_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': feature_values
                })
                st.dataframe(sample_df)
            
            with col2:
                # Show true and predicted class with enhanced visuals
                st.markdown("### Classification Results")
                
                # Get true label
                true_class = st.session_state.data_info['label_names'][sample_label.item()]
                
                # Make prediction
                with torch.no_grad():
                    output = st.session_state.model(sample_datapoint.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probs).item()
                    predicted_class = st.session_state.data_info['label_names'][prediction]
                
                # Create a more visually appealing result display
                is_correct = prediction == sample_label.item()
                result_color = "#28a745" if is_correct else "#dc3545"
                icon = "✅" if is_correct else "❌"
                
                st.markdown(f"""
                <div style="background-color: rgba({','.join(['40, 167, 69, 0.1' if is_correct else '220, 53, 69, 0.1'])});
                     padding: 15px; border-radius: 10px; border-left: 5px solid {result_color};">
                    <h4>Prediction Result {icon}</h4>
                    <p><b>True Class:</b> {true_class}</p>
                    <p><b>Predicted Class:</b> {predicted_class}</p>
                    <p><b>Confidence:</b> {probs[prediction].item()*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show class-specific visuals based on dataset
                if dataset_name == "Iris":
                    # Display the main iris image instead of individual species images
                    st.markdown("### Iris Classification")
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                        <p>Predicted class: <b>{predicted_class}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Display the iris image in full width
                    st.image("assets/iris.png", caption="Iris Species Comparison", use_column_width=True)
                elif dataset_name == "Titanic":
                    # Display the Titanic image
                    st.markdown("### Titanic Passenger Classification")
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                        <p>Predicted survival status: <b>{predicted_class}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Display the Titanic image in full width
                    st.image("assets/titanic.jpg", caption="RMS Titanic", use_column_width=True)
                
                # Show probabilities as a horizontal bar chart
                st.markdown("### Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': st.session_state.data_info['label_names'],
                    'Probability': probs.numpy()
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                            color='Probability', color_continuous_scale='Blues',
                            text=prob_df['Probability'].apply(lambda x: f"{x:.2%}"))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display activation path visualization using Plotly (interactive)
            st.subheader("Activation Path")
            
            # Use the activation_tracer for interactive visualization
            if 'activation_tracer' not in st.session_state:
                # Create one if it doesn't exist
                st.session_state.activation_tracer = ActivationTracer(
                    st.session_state.model,
                    st.session_state.data_info['feature_names'],
                    st.session_state.data_info['label_names']
                )
            
            # Get and display the interactive visualization
            fig = st.session_state.activation_tracer.create_interactive_network_flow(
                sample_input=sample_datapoint,
                sample_label=sample_label
            )
            
            # Make sure to use st.plotly_chart for Plotly figures
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional activation visualizations (optional)
            if st.checkbox("Show Detailed Neuron Activation Heatmap"):
                st.subheader("Neuron Activation Heatmap")
                fig = st.session_state.visualizer.visualize_activations(sample_datapoint)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please train a model and select a sample to see activations.")

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