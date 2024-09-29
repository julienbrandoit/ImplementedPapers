# Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

This folder contains the first part of my implementation of *Physics Informed Neural Networks (PINNs)*, based on the work presented in the paper *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations* by M. Raissi, P. Perdikaris, and G. E. Karniadakis. This implementation focuses on the continuous time approach for solving nonlinear PDEs.

## Overview

The goal of this project is to use deep learning models to solve nonlinear partial differential equations (PDEs) in a data-driven manner. The idea behind PINNs is to encode the governing physical laws of the system directly into the neural network architecture by incorporating the PDE as a regularization term during training. This allows the network to find solutions that not only fit the data but also respect the physical properties of the system.

### Part I: Continuous Time Implementation

This repository currently covers the continuous time approach for solving PDEs. Specifically, I have implemented and extended two key examples:

1. **Burgers' Equation (1D)**: This example is based on the original problem presented in the paper, and the results closely match the exact solutions.
2. **Heat Equation (1D)**: A personal experiment that applies the PINN framework to the heat equation, with an error analysis comparing the model’s predictions to the exact analytical solution.

I plan to explore discrete time approaches and the part II of the article in the future.

## Notebooks

### 1. Burgers' Equation (1D)

- **File**: `pinn_burgers1D.ipynb`
- **Description**: This notebook implements a physics-informed neural network for solving the 1D Burgers' equation. It follows the methodology outlined in the original paper, using automatic differentiation to compute the PDE residual and incorporating it into the loss function. The results demonstrate that the PINN effectively captures the behavior of the exact solution.

### 2. Heat Equation (1D)

- **File**: `pinn_heat.ipynb`
- **Description**: This notebook applies the PINN framework to solve the 1D heat equation. The model is trained on synthetic data generated from the exact solution, and I perform error analysis to compare the network’s predictions to the exact analytical solution. The results are promising, with the network achieving a low error and closely approximating the true solution.

## Results

- **Burgers' Equation (1D)**: The neural network successfully learned the solution to the Burgers' equation with results that closely match the exact solution.
- **Heat Equation (1D)**: The model achieves a good approximation of the heat equation solution, with error measurements from the exact analytical solution validating the accuracy of the predictions. I also explain how to derive the exact analytical solution thanks to variables separations and Fourier decomposition of the IC.

## Future Work

In the future, I plan to extend this implementation by exploring the following areas:

- **Part II**
- **Discrete Time Approach**: Implementing the discrete time formulation of PINNs to investigate its performance on temporal data.

# Dev's note

From an engineering perspective - especially a Bayesian one - I have a strong interest in how to incorporate prior knowledge into methods, particularly in deep learning, which is inherently data-driven rather than knowledge-driven. Physics-Informed Neural Networks (PINNs) are an ingenious approach to achieving this when a mathematical description is available.

- I forgot that the residuals are evaluated at the collocation points, while the target values are applied at the $N_u$ points (the training set).


---

This is a work in progress, and I plan to explore more of PINNs.
