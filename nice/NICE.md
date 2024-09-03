# NICE Implementation

This folder contains the implementation of the NICE (Non-linear Independent Components Estimation) model. It includes two Jupyter notebooks demonstrating the model’s application on different datasets.

## Overview

The NICE model, as introduced in the paper "Non-linear Independent Components Estimation," is implemented here with two main experiments:

1. **MNIST Experiment**: Reproduces the results of NICE on the MNIST dataset.
2. **Proof of Concept**: Applies NICE to a synthetic sinusoidal 2D distribution to validate the model’s capabilities in a controlled setting.

## Notebooks

### 1. MNIST Experiment

- **File**: `nice_mnist.ipynb`
- **Description**: This notebook implements the NICE model on the MNIST dataset. It follows almost exactly the experimental setup described in the NICE paper and achieves a log-likelihood of 2146.8961, similar to the results reported in the original paper.
- **Usage**: To run this notebook, ensure you have the necessary dependencies installed and execute the cells to train the model and evaluate its performance on MNIST.

### 2. Proof of Concept

- **File**: `nice_sin.ipynb`
- **Description**: This notebook serves as a proof of concept for NICE by applying it to a synthetic sinusoidal 2D distribution. This experiment helps in understanding the model’s behavior and performance on simple, generated data.
- **Usage**: Run this notebook to see how NICE models the sinusoidal distribution and evaluates its ability to handle synthetic, non-trivial data.

## Results

- **MNIST Experiment**: The model achieves a log-likelihood of 2146.8961, demonstrating comparable performance to the results presented in the NICE paper.
- **Proof of Concept**: The notebook includes visualizations and metrics that illustrate how the model captures the structure of the sinusoidal distribution.
