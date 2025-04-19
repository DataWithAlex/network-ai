# Neural Network Visualizer

## A Graph-Theoretical Approach to Neural Network Interpretability

This application provides an interactive framework for visualizing and interpreting neural networks through the lens of graph theory, addressing the "black box" problem in machine learning by making the internal workings of neural networks more transparent and interpretable.

## Overview

The Neural Network Visualizer is a Streamlit-based application that enables users to:

- Construct, train, and visualize multi-layer perceptron (MLP) neural networks
- Explore network architecture through interactive graph visualizations
- Analyze data transformations at each layer of the network
- Investigate feature importance and contributions to model decisions
- Visualize high-dimensional data projections using PCA and t-SNE
- Examine neuron activations for specific data points with interactive flow diagrams
- Evaluate model performance with comprehensive metrics and visualizations

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/neural-network-visualizer.git
   cd neural-network-visualizer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch the Streamlit application:
```bash
streamlit run app.py
```

Navigate to the URL displayed in the terminal (typically http://localhost:8501).

## Technical Architecture

### Project Structure
```
neural-network-visualizer/
├── app.py                      # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── activation_tracer.py    # Specialized activation tracing and visualization
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── interactive_visualizer.py # Interactive Plotly visualizations
│   ├── model_builder.py        # Neural network model definitions
│   ├── trainer.py              # Training and evaluation functions
│   └── visualizer.py           # Main visualization coordinator
└── requirements.txt            # Dependencies
```

### Module Descriptions

#### `app.py`
The main Streamlit application that provides the user interface and orchestrates the interaction between different modules. It handles:
- User input for dataset selection and model configuration
- Model training and evaluation
- Visualization rendering and display
- Session state management

#### `modules/activation_tracer.py`
Specialized class for tracing activations through neural networks and creating interactive visualizations of activation flow. Features:
- Traces sample inputs through the network
- Creates interactive network flow visualizations with Plotly
- Visualizes activation values at each layer
- Highlights important connections based on weight values

#### `modules/data_loader.py`
Handles dataset loading, preprocessing, and batching. Includes:
- Base dataset class with common functionality
- Specialized dataset classes for Iris and Titanic datasets
- Data normalization and train/test splitting
- PyTorch DataLoader integration

#### `modules/interactive_visualizer.py`
Creates interactive network visualizations using Plotly. Features:
- Interactive node and edge representations
- Hover information for network components
- Color-coding for different layer types
- Customizable layout options

#### `modules/model_builder.py`
Defines neural network architectures and provides factory methods for model creation. Includes:
- Base model class with common functionality
- MLP classifier implementation
- Activation hooks for capturing intermediate outputs
- Methods for extracting model structure and weights

#### `modules/trainer.py`
Handles model training, evaluation, and performance metrics. Features:
- Training loop with epoch-level control
- Validation during training
- Comprehensive performance metrics
- Training history tracking

#### `modules/visualizer.py`
Main visualization coordinator that delegates to specialized visualizers. Includes:
- Network architecture visualization
- Feature importance analysis
- Data projection visualization (PCA, t-SNE)
- Activation visualization

## Features

### 1. Dataset Selection and Preprocessing
- **Supported Datasets**: Iris (flower classification) and Titanic (survival prediction)
- **Preprocessing**: Automatic normalization, encoding, and train/test splitting
- **Extensible Framework**: Modular design allows easy addition of new datasets

### 2. Model Architecture Configuration
- **Layer Configuration**: Customize the number and size of hidden layers
- **Hyperparameter Tuning**: Adjust learning rate, batch size, epochs, and dropout rate
- **Interactive Controls**: Sliders and selectors for parameter adjustment

### 3. Network Architecture Visualization
- **Interactive Graph**: Plotly-based visualization of network structure
- **Layer Representation**: Color-coded nodes for different layer types (input, hidden, output)
- **Weight Visualization**: Edge thickness and color represent weight magnitude and sign
- **Hover Information**: Detailed information on hover for nodes and edges

### 4. Training Process Visualization
- **Learning Curves**: Real-time plotting of loss and accuracy during training
- **Validation Metrics**: Simultaneous visualization of training and validation performance
- **Convergence Analysis**: Visual assessment of model convergence and potential overfitting

### 5. Performance Evaluation
- **Confusion Matrix**: Interactive visualization of classification results
- **Classification Report**: Precision, recall, and F1-score for each class
- **Overall Metrics**: Accuracy and other aggregate performance measures

### 6. Feature Importance Analysis
- **Weight-based Importance**: Visualization of feature importance based on network weights
- **Contribution Analysis**: Assessment of each feature's contribution to model decisions
- **Comparative Visualization**: Ranking of features by importance

### 7. Data Projections
- **PCA Visualization**: Linear dimensionality reduction to visualize data distribution
- **t-SNE Visualization**: Non-linear dimensionality reduction for complex relationships
- **Layer-wise Projections**: Visualization of data transformations through network layers
- **Class Separation**: Visual assessment of class separability at different layers

### 8. Activation Visualization
- **Interactive Flow Diagram**: Visualization of activation flow through the network
- **Neuron Activation Values**: Color-coded representation of activation strengths
- **Sample-specific Analysis**: Examine network behavior for specific input examples
- **Prediction Explanation**: Connect activation patterns to model predictions

## Technical Implementation Details

### Neural Network Implementation
- **Framework**: PyTorch for neural network implementation
- **Architecture**: Modular design with customizable layer configurations
- **Activation Functions**: ReLU for hidden layers, softmax for output layer
- **Optimization**: Adam optimizer with configurable learning rate
- **Loss Function**: Cross-entropy loss for classification tasks

### Visualization Technologies
- **Interactive UI**: Streamlit for web interface and controls
- **Graph Visualization**: Plotly for interactive network visualizations
- **Statistical Plots**: Matplotlib and Plotly for performance metrics
- **Dimensionality Reduction**: Scikit-learn for PCA and t-SNE implementations

### Data Processing
- **Data Handling**: Pandas for data manipulation and preprocessing
- **Numerical Processing**: NumPy for efficient numerical operations
- **Batching**: PyTorch DataLoader for efficient batch processing
- **Normalization**: StandardScaler for feature normalization

## Research Applications

This tool is designed to support research in several areas:

1. **Neural Network Interpretability**: Investigate how neural networks transform data and make decisions
2. **Feature Engineering**: Identify important features and their interactions
3. **Model Comparison**: Compare different architectures and their internal representations
4. **Educational Purposes**: Teach neural network concepts through interactive visualization
5. **Hyperparameter Optimization**: Visually assess the impact of different hyperparameters

## Limitations and Future Work

### Current Limitations
- Limited to MLP architectures (no CNN, RNN support yet)
- Supports only classification tasks
- Limited to tabular datasets
- No support for very large networks due to visualization constraints

### Planned Enhancements
- Support for convolutional neural networks (CNNs)
- Visualization of recurrent neural networks (RNNs)
- Integration of additional explainable AI techniques (SHAP, LIME)
- Support for custom dataset upload
- Export of visualizations as interactive HTML files
- Comparative visualization of multiple models
- Adversarial example generation and visualization
- Integration with pre-trained models from popular frameworks

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Streamlit 1.0+
- Plotly 5.0+
- Scikit-learn 1.0+
- Pandas 1.3+
- NumPy 1.20+
- Matplotlib 3.4+
- NetworkX 2.6+

See `requirements.txt` for specific version requirements.

## Citation

If you use this tool in your research, please cite:

```
@software{neural_network_visualizer,
  author = {Your Name},
  title = {Neural Network Visualizer: A Graph-Theoretical Approach to Neural Network Interpretability},
  year = {2023},
  url = {https://github.com/your-username/neural-network-visualizer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Streamlit team for their excellent framework for building data applications
- The PyTorch community for their comprehensive deep learning library
- The scikit-learn team for their machine learning and visualization tools