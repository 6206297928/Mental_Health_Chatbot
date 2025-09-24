# 3D Visualization of Linear and Non-linear Neurons

This repository demonstrates the difference between **linear** and **non-linear neuron activations** using Python and Matplotlib. The 3D surface plots help visualize how different activation functions transform a linear combination of inputs.

---

## Features

- **Linear Neuron**:  
  \[
  y = w_1 x_1 + w_2 x_2 + b
  \]

- **Non-linear Neurons** with popular activation functions:  
  - **ReLU**: \( \text{ReLU}(x) = \max(0, x) \)  
  - **Sigmoid**: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)  
  - **Tanh**: \( \tanh(x) \)

- 3D surface plots for visual comparison of linear vs non-linear outputs.

---

## Requirements

- Python 3.x  
- Numpy  
- Matplotlib  

You can install the required libraries using:

```bash
pip install numpy matplotlib
