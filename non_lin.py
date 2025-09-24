import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define input space
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# Weights and bias
w1, w2, b = 1.0, -1.0, 0.5
Z_linear = w1*X1 + w2*X2 + b

# Activation functions
Z_relu = np.maximum(0, Z_linear)
Z_sigmoid = 1 / (1 + np.exp(-Z_linear))
Z_tanh = np.tanh(Z_linear)

# Plot function
def plot_surface(X1, X2, Z, title):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Output')
    fig.colorbar(surf)
    plt.show()

# Visualizations
plot_surface(X1, X2, Z_linear, "Linear Neuron: y = w1*x1 + w2*x2 + b")
plot_surface(X1, X2, Z_relu, "Non-linear Neuron: ReLU(w1*x1 + w2*x2 + b)")
plot_surface(X1, X2, Z_sigmoid, "Non-linear Neuron: Sigmoid(w1*x1 + w2*x2 + b)")
plot_surface(X1, X2, Z_tanh, "Non-linear Neuron: Tanh(w1*x1 + w2*x2 + b)")
