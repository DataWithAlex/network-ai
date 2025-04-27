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

# Create an expander for model configuration
with st.expander("Model Configuration", expanded=True):
    # Create three columns for the controls
    col1, col2, col3 = st.columns(3)

    # Dataset and model selection in first column
    with col1:
        st.subheader("Dataset & Model")
        
        # Dataset selection
        dataset_name = st.selectbox(
            "Select Dataset",
            ["Iris 🪻", "Titanic 🚢"]
        )
        
        # Model type selection
        model_type = st.selectbox(
            "Select Model Type",
            ["MLP 💾"]
        )

    # Model architecture in second column
    with col2:
        st.subheader("Model Architecture")
        
        # Set default hidden sizes based on dataset
        if dataset_name == "Iris 🪻":
            default_hidden_sizes = [8, 4]
        else:
            default_hidden_sizes = [16, 8]
        
        hidden_layer_1 = st.slider("Neurons in Hidden Layer 1", 2, 32, default_hidden_sizes[0])
        hidden_layer_2 = st.slider("Neurons in Hidden Layer 2", 2, 32, default_hidden_sizes[1])
        hidden_sizes = [hidden_layer_1, hidden_layer_2]
        
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.0, 0.1)

    # Training parameters in third column
    with col3:
        st.subheader("Training Parameters")
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            value=0.001
        )
        
        epochs = st.slider("Number of Epochs", 10, 200, 50)
        batch_size = st.slider("Batch Size", 8, 128, 32)

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
    # Clean dataset name by removing emojis for the data loader
    clean_dataset_name = "Iris" if "Iris" in dataset_name else "Titanic"
    
    # Load and preprocess dataset
    dataset = get_dataset(clean_dataset_name, batch_size=batch_size)
    data_info = dataset.load_data()
    
    # Create model
    model = create_model(
        model_type=model_type.split()[0],  # Remove emoji from model type
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

# Train model button (centered)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            load_and_train()
        st.success("Model trained successfully!")

# Tabs for different visualizations
if st.session_state.model is not None:
    # Remove the Activation Visualization tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Information",
        "Network Architecture", 
        "Training History", 
        "Feature Importance", 
        "Data Projections"
    ])
    
    # Tab 1: Model Information
    with tab1:
        st.header("Model Information")
        
        # Dataset Information
        st.subheader("Dataset Information")
        
        # Display dataset image centered above text
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if dataset_name == "Iris 🪻":
                st.image("assets/iris.png", caption="Iris Dataset", use_column_width=True)
            elif dataset_name == "Titanic":
                st.image("assets/titanic.jpg", caption="Titanic Dataset", use_column_width=True)
        
        # Dataset description
        if dataset_name == "Iris 🪻":
            st.write("""
            The Iris dataset consists of 150 samples from three species of Iris flowers. Each sample has four features: sepal length, 
            sepal width, petal length, and petal width. This is a classic dataset for classification tasks.
            """)
        elif dataset_name == "Titanic":
            st.write("""
            The Titanic dataset contains information about passengers aboard the Titanic, including whether they survived. 
            Features include age, gender, passenger class, fare, and more. This dataset is commonly used for binary classification.
            """)
        
        # Dataset statistics
        st.markdown("**Dataset Statistics:**")
        
        # Get dataset info from session state
        if 'data_info' in st.session_state:
            data_info = st.session_state.data_info
            
            # Create bullet points for dataset statistics
            st.markdown(f"* Number of features: {data_info['n_features']}")
            st.markdown(f"* Number of classes: {data_info['n_classes']}")
            st.markdown(f"* Feature names: {', '.join(data_info['feature_names'])}")
            st.markdown(f"* Class names: {', '.join(data_info['label_names'])}")
        
        # Model Hyperparameters
        st.subheader("Model Hyperparameters")
        
        if 'model' in st.session_state:
            hyperparams = st.session_state.model.get_hyperparameters()
            
            # Create a dataframe for hyperparameters
            hyperparams_df = pd.DataFrame({
                'Parameter': list(hyperparams.keys()),
                'Value': list(hyperparams.values())
            })
            
            st.dataframe(hyperparams_df)
        
        # Model Description
        st.subheader("Model Description")
        
        if 'model' in st.session_state:
            st.write(st.session_state.model.get_description())
    
    # Tab 2: Network Architecture
    with tab2:
        st.header("Neural Network Architecture")
        st.markdown("""
        This interactive visualization shows the complete architecture of the neural network and how data flows through it.
        
        ### How to Use:
        1. **Hover** over any node to see its details and activation values
        2. **Observe the connections** between layers to understand data flow
        3. **Note the color and thickness** of connections to see weight importance
        """)
        
        # Initialize visualizer if not already done
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = NetworkVisualizer(
                st.session_state.model,
                st.session_state.data_info['feature_names'],
                st.session_state.data_info['label_names']
            )
        
        # Initialize activation tracer if not already done
        if 'activation_tracer' not in st.session_state:
            st.session_state.activation_tracer = ActivationTracer(
                st.session_state.model,
                st.session_state.data_info['feature_names'],
                st.session_state.data_info['label_names']
            )
        
        # Display the static network architecture diagram
        st.subheader("Network Architecture Diagram")
        st.markdown("""
        This diagram shows the complete structure of the neural network including:
        - **Input Layer**: Features from the dataset
        - **Hidden Layers**: Fully connected layers with their activation functions
        - **Output Layer**: Class probabilities
        
        Hover over nodes to see details about each layer.
        """)
        
        # Get the network architecture visualization
        arch_fig = st.session_state.visualizer.plot_network_interactive()
        st.plotly_chart(arch_fig, use_container_width=True)
        
        # Sample selection for activation flow
        st.subheader("Activation Flow Visualization")
        st.markdown("""
        Select a sample to see how data flows through the network. This helps understand how the model processes 
        inputs to make predictions.
        """)
        
        if st.button("Select Random Sample"):
            data_info = st.session_state.data_info
            test_dataset = data_info["test_dataset"]
            sample_idx = np.random.randint(0, len(test_dataset))
            st.session_state.sample_datapoint, st.session_state.sample_label = test_dataset[sample_idx]
        
        # Display the sample details and network flow
        if hasattr(st.session_state, 'sample_datapoint') and st.session_state.sample_datapoint is not None:
            sample_datapoint = st.session_state.sample_datapoint
            sample_label = st.session_state.sample_label
            
            # Display sample information first
            st.markdown("### Input Features")
            
            # Create a more visually appealing feature display
            feature_names = st.session_state.data_info['feature_names']
            feature_values = sample_datapoint.numpy()
            
            # Create columns for feature visualization
            feature_cols = st.columns(len(feature_names))
            
            # Display each feature in its own styled container
            for i, (name, value) in enumerate(zip(feature_names, feature_values)):
                with feature_cols[i]:
                    st.metric(
                        label=name,
                        value=f"{value:.4f}",
                        delta=None
                    )
            
            # Display true and predicted labels
            with torch.no_grad():
                output = st.session_state.model(sample_datapoint.unsqueeze(0))
                probs = torch.softmax(output, dim=1)[0]
                prediction = torch.argmax(probs).item()
            
            true_label = st.session_state.data_info['label_names'][sample_label]
            pred_label = st.session_state.data_info['label_names'][prediction]
            
            # Create a visual indicator for prediction correctness
            is_correct = prediction == sample_label
            
            # Display prediction result with visual styling
            if is_correct:
                st.success(f"✅ TRUE POSITIVE: Model correctly predicted {pred_label}")
            else:
                st.error(f"❌ FALSE PREDICTION: Model predicted {pred_label}, but true class is {true_label}")
            
            # Create two columns for true and predicted class
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**True Class:** {true_label}")
            with col2:
                if is_correct:
                    st.success(f"**Predicted Class:** {pred_label}")
                else:
                    st.error(f"**Predicted Class:** {pred_label}")
            
            # Display prediction probabilities as a horizontal bar chart
            st.markdown("### Prediction Probabilities")
            
            # Create a more visual probability display
            fig = go.Figure()
            
            # Add bars for each class probability
            for i, (label, prob) in enumerate(zip(st.session_state.data_info['label_names'], probs.numpy())):
                # Highlight the predicted class
                color = 'rgba(0, 204, 102, 0.8)' if i == prediction else 'rgba(49, 130, 189, 0.7)'
                
                # Add bar with percentage
                fig.add_trace(go.Bar(
                    x=[prob],
                    y=[label],
                    orientation='h',
                    text=[f"{prob:.2%}"],
                    textposition='auto',
                    marker_color=color,
                    name=label
                ))
            
            # Update layout
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability",
                yaxis_title="Class",
                xaxis=dict(range=[0, 1]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the activation path visualization AFTER the tables
            st.markdown("### Network Activation Flow")
            
            # Get and display the interactive visualization
            flow_fig = st.session_state.activation_tracer.create_interactive_network_flow(
                sample_input=sample_datapoint,
                sample_label=sample_label
            )
            
            # Make sure to use st.plotly_chart for Plotly figures
            st.plotly_chart(flow_fig, use_container_width=True)
            
            # Add explanation of the visualization
            with st.expander("How to interpret this visualization"):
                st.markdown("""
                ### Understanding the Network Flow
                
                - **Input nodes** (left): Show the normalized feature values
                - **Hidden layer nodes** (middle): Show activation values after processing
                - **Output nodes** (right): Show the probability for each class
                
                **Colors indicate:**
                - **Green connections**: Positive weights (activating)
                - **Red connections**: Negative weights (inhibiting)
                - **Thickness**: Magnitude of the weight (stronger influence)
                
                **Hover** over any node or connection to see more details!
                """)
        else:
            # If no sample is selected, show a message
            st.info("Select a sample to visualize data flow through the network.")
        
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
            
            # Get label names and confusion matrix
            label_names = st.session_state.data_info['label_names']
            conf_matrix = metrics['confusion_matrix']
            
            # Create a better confusion matrix visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use a more appealing colormap
            cmap = plt.cm.Blues
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
            
            # Add title and labels
            ax.set_title('Confusion Matrix', fontsize=16)
            ax.set_xlabel('Predicted Label', fontsize=14)
            ax.set_ylabel('True Label', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Count', rotation=270, labelpad=15, fontsize=12)
            
            # Add tick marks and class names
            tick_marks = np.arange(len(label_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(label_names, fontsize=12, rotation=45, ha='right')
            ax.set_yticklabels(label_names, fontsize=12)
            
            # Add text annotations with counts and percentages
            thresh = conf_matrix.max() / 2.0
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    value = conf_matrix[i, j]
                    total = conf_matrix[i].sum()
                    percentage = value / total * 100 if total > 0 else 0
                    
                    # Format text to show count and percentage
                    text = f"{value}\n({percentage:.1f}%)"
                    
                    # Choose text color based on background darkness
                    color = "white" if conf_matrix[i, j] > thresh else "black"
                    
                    ax.text(j, i, text, ha="center", va="center", 
                           color=color, fontsize=12, fontweight="bold")
            
            # Add grid lines for better readability
            ax.set_xticks(np.arange(-.5, len(label_names), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(label_names), 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
            
            # Tight layout to ensure everything fits
            plt.tight_layout()
            
            # Display the figure
            st.pyplot(fig)
            
            # Add explanation of confusion matrix
            with st.expander("How to interpret the confusion matrix"):
                st.markdown("""
                ### Understanding the Confusion Matrix
                
                The confusion matrix shows how well the model is classifying each class:
                
                - **Rows**: True labels (actual classes)
                - **Columns**: Predicted labels (what the model predicted)
                - **Diagonal elements**: Correctly classified instances
                - **Off-diagonal elements**: Misclassified instances
                
                Each cell shows:
                - The **count** of instances
                - The **percentage** of the true class
                
                A perfect model would have all instances on the diagonal.
                """)
        
        with col2:
            st.subheader("ROC Curve")
            
            # Get predictions and true labels for ROC curve
            y_true = metrics.get('y_true', [])
            y_score = metrics.get('y_score', [])
            
            # Check if we have the necessary data for ROC curve
            if len(y_true) > 0 and len(y_score) > 0:
                # Create ROC curve
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # For binary classification
                if len(label_names) == 2:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=14)
                    ax.set_ylabel('True Positive Rate', fontsize=14)
                    ax.set_title('Receiver Operating Characteristic', fontsize=16)
                    ax.legend(loc="lower right")
                
                # For multi-class classification
                else:
                    from sklearn.metrics import roc_curve, auc
                    from sklearn.preprocessing import label_binarize
                    
                    # Binarize the labels for multi-class ROC
                    classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=classes)
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i, class_name in enumerate(label_names):
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        # Plot ROC curve for each class
                        # Use a colormap that's compatible with matplotlib versions
                        color = plt.cm.tab10(i % 10)
                        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                                label=f'{class_name} (area = {roc_auc[i]:.2f})')
                    
                    ax.plot([0, 1], [0, 1], 'k--', lw=2)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=14)
                    ax.set_ylabel('True Positive Rate', fontsize=14)
                    ax.set_title('Multi-class ROC', fontsize=16)
                    ax.legend(loc="lower right")
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("ROC curve data not available. Make sure to include prediction scores in the metrics.")
        
        # Add a single expander for both visualizations
        with st.expander("How to interpret these visualizations"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Understanding the Confusion Matrix
                
                The confusion matrix shows how well the model is classifying each class:
                
                - **Rows**: True labels (actual classes)
                - **Columns**: Predicted labels (what the model predicted)
                - **Diagonal elements**: Correctly classified instances
                - **Off-diagonal elements**: Misclassified instances
                
                Each cell shows:
                - The **count** of instances
                - The **percentage** of the true class
                
                A perfect model would have all instances on the diagonal.
                """)
            
            with col2:
                st.markdown("""
                ### Understanding the ROC Curve
                
                The ROC curve plots the True Positive Rate against the False Positive Rate:
                
                - **True Positive Rate (Sensitivity)**: Proportion of actual positives correctly identified
                - **False Positive Rate (1-Specificity)**: Proportion of actual negatives incorrectly identified
                
                The **Area Under the Curve (AUC)**:
                - 1.0 = Perfect classifier
                - 0.5 = No better than random guessing (diagonal line)
                - Higher AUC indicates better model performance
                """)
        
        # Classification Report (fixed to handle missing dict)
        st.subheader("Classification Report")
        
        # Check if classification_report_dict exists, otherwise use the text version
        if 'classification_report_dict' in metrics:
            report_df = pd.DataFrame(metrics['classification_report_dict']).transpose()
            
            # Drop unnecessary rows
            if 'accuracy' in report_df.index:
                accuracy_row = report_df.loc[['accuracy']]
                report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
                
                # Reorder columns for better readability
                if 'support' in report_df.columns:
                    cols = ['precision', 'recall', 'f1-score', 'support']
                    report_df = report_df[cols]
                
                # Format the values
                for col in ['precision', 'recall', 'f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].map(lambda x: f"{x:.2f}")
                
                # Display the report as a styled dataframe
                st.dataframe(report_df, use_container_width=True)
                
                # Display accuracy separately
                if 'accuracy' in accuracy_row.columns:
                    st.metric("Overall Accuracy", f"{accuracy_row['accuracy'].values[0]:.2f}")
        else:
            # Fallback to text display if dataframe conversion fails
            if 'classification_report' in metrics:
                st.text(metrics['classification_report'])
            else:
                st.info("Detailed classification report not available.")
    
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