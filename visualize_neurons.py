# visualize_neurons.py
"""
3D Visualization of Linear and Non-linear Neurons

This script demonstrates the difference between a linear neuron and
non-linear neurons using ReLU, Sigmoid, and Tanh activation functions.
It generates 3D surface plots for visual comparison.

Author: Sukumar Poddar
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# ----------------------------
# Define input space
# ----------------------------
x1 = np.linspace(-5, 5, 100)  # Input 1 range
x2 = np.linspace(-5, 5, 100)  # Input 2 range
X1, X2 = np.meshgrid(x1, x2)  # Create a 2D grid

# ----------------------------
# Define weights and bias
# ----------------------------
w1, w2, b = 1.0, -1.0, 0.5  # Example weights and bias

# Linear combination
Z_linear = w1*X1 + w2*X2 + b

# ----------------------------
# Apply activation functions
# ----------------------------
Z_relu = np.maximum(0, Z_linear)            # ReLU
Z_sigmoid = 1 / (1 + np.exp(-Z_linear))    # Sigmoid
Z_tanh = np.tanh(Z_linear)                  # Tanh

# ----------------------------
# Plotting function
# ----------------------------
def plot_surface(X1, X2, Z, title):
    """
    Plots a 3D surface plot of the given data.
    
    Parameters:
    - X1, X2: Input grids
    - Z: Output values
    - title: Title of the plot
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Output')
    fig.colorbar(surf)
    plt.show()

# ----------------------------
# Generate visualizations
# ----------------------------
plot_surface(X1, X2, Z_linear, "Linear Neuron: y = w1*x1 + w2*x2 + b")
plot_surface(X1, X2, Z_relu, "Non-linear Neuron: ReLU(w1*x1 + w2*x2 + b)")
plot_surface(X1, X2, Z_sigmoid, "Non-linear Neuron: Sigmoid(w1*x1 + w2*x2 + b)")
plot_surface(X1, X2, Z_tanh, "Non-linear Neuron: Tanh(w1*x1 + w2*x2 + b)")
