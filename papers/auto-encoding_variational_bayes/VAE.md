# VAE Implementation

This folder contains the implementation of the VAE model. It includes three Jupyter notebooks that demonstrate the model’s application.

## Overview

The VAE model, as introduced in the paper *Auto-Encoding Variational Bayes*, is implemented here with two main experiments:

1. **MNIST Experiment**: Reproduces the results of VAE on the MNIST dataset using a 20-D latent space.
 
2. **Optimizer Comparison Experiment**: Compares the score of the Adagrad optimizer (originally used in the paper) with the score of the Adam optimizer.
 
3. **Latent Space Exploration**: Explores the learned latent space by linear interpolation and decoding.
 
## Notebooks

### 1. 20-D MNIST Experiment

- **Files**: `vae_mnist_training-adam.ipynb` and `vae_mnist_training-adagrad`
- **Description**: Those notebooks implement the VAE architecture introduced in the paper (assuming a Bernoulli distribution of the data) and train it using either adagrad, as in the original paper, or adam. I encode data in a 20-D latent space.

### 2. **Latent Space Exploration Experiment**:

-   **File**: `vae_mnist_interpolating.ipynb`
-   **Description**: This notebook essentially loads the trained VAE, computes the test and validation scores and explore the learned latent space by linear interpolation between encoded images from the test set. The goal is to get an idea of how the latent space is structured by the VAE during the training phase.

## Results

I achieve result that are very similar to those of the paper (using both adagrad and adam) : a score of $\approx$ 100 for the lower-bound using a 20-D latent space and 500 hidden units. The generated sample are not of very good quality but this is also the case in the original paper. I really think that introducing a hidden layer in the NN would increase *a lot* the quality of the encoding/decoding.

There is not a lot of difference between the final scores obtain with Adagrad in comparison to Adam.

The interpolation in the latent space shows a well structured space with almost all the intermediate values being meaningful once decoded. Some VAE interpretations explain that this is because of the regularization term introduced by the KL divergence between the prior and the encoder. 

# Dev's note

- Implementing a VAE is quite straightforward. However I really want to explore more complex case in the future and redo the mathematical derivation of the analytical result of the KL divergence between some distribution.

- I should explore how to get an estimation of the likelihood of the data. 

- I really don't understand why the NN used in the paper are so "simple" (no hidden layer).