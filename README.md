# Neural Network Visualizer

A comprehensive Streamlit application for visualizing neural networks through the lens of graph theory, making the internal workings of neural networks more transparent and addressing the "black box" problem.

## Overview

This application allows users to:
- Visualize neural network architectures as interactive graphs
- Understand data transformations at each layer of the network
- Explore feature importance in the model
- Visualize data projections using PCA and t-SNE
- Examine neuron activations for specific data points

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/neural-network-visualizer.git
   cd neural-network-visualizer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

Then open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

## Features

### 1. Dataset Selection
- Choose between Iris and Titanic datasets
- Modular design allows easy extension to other datasets

### 2. Model Architecture Configuration
- Customize the number of neurons in hidden layers
- Adjust learning rate, batch size, and number of epochs

### 3. Network Architecture Visualization
- Static visualization using NetworkX
- Interactive visualization using Plotly
- Color-coded nodes and edges to represent different layers and weight magnitudes

### 4. Training History
- Loss and accuracy curves during training
- Confusion matrix visualization
- Detailed classification report

### 5. Feature Importance Analysis
- Bar chart visualization of feature importance based on network weights
- Helps identify which input features contribute most to the model's decisions

### 6. Data Projections
- PCA projections showing how data is distributed in 2D space
- t-SNE projections for non-linear dimensionality reduction
- Layer-wise projections showing how data representations evolve through the network

### 7. Activation Visualization
- Visualize neuron activations for specific input examples
- Examine how information flows through the network
- Prediction probabilities for classification tasks

## Implementation Details

### Project Structure
```
nn_visualizer/
├── app.py                  # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── model_builder.py    # Neural network model definitions  
│   ├── trainer.py          # Training and evaluation functions
│   └── visualizer.py       # Visualization functions
└── requirements.txt        # Dependencies
```

### Technologies Used
- Python 3.13.1
- PyTorch 2.2.2 for neural network implementation
- Streamlit for the web interface
- NetworkX and Plotly for graph visualization
- Scikit-learn for data processing and dimensionality reduction
- Matplotlib for static visualizations
- Pandas for data manipulation

## Example Visualizations

The application provides several types of visualizations:

1. **Network Architecture**: Visualize the neural network as a graph, with nodes representing neurons and edges representing weights.

2. **Training Metrics**: Track loss and accuracy during training to understand model convergence.

3. **Feature Importance**: Identify which features have the strongest influence on the model's predictions.

4. **Data Projections**: See how data is transformed and separated at each layer of the network.

5. **Neuron Activations**: Observe how individual neurons respond to specific inputs.

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Future Improvements

- Support for convolutional neural networks (CNNs)
- Visualization of recurrent neural networks (RNNs)
- Additional datasets and pre-trained models
- Explainable AI techniques like SHAP values and LIME
- Export visualizations as interactive HTML files
- Support for custom dataset upload

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.