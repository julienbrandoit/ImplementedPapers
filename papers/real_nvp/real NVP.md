# Real NVP Implementation

This folder contains the implementation of the Real NVP (Real-valued Non-Volume Preserving transformations) model. It includes two Jupyter notebooks that demonstrate the model’s application on different datasets.

## Overview

The Real NVP model, as introduced in the paper *Density Estimation using Real NVP*, is implemented here with two main experiments:

1. **CelebA Experiment**: Reproduces the results of Real NVP on the CelebA dataset, using a subset of aligned and cropped images.
2. **Proof of Concept**: Applies Real NVP to a synthetic sinusoidal 2D distribution to validate the model’s capabilities in a controlled setting.

## Notebooks

### 1. CelebA Experiment

- **File**: `real_nvp_celeba.ipynb`
- **Description**: This notebook implements the Real NVP model on the CelebA dataset. It follows closely the experimental setup described in the original paper, implementing specific techniques such as batch normalization, weight normalization, and the scaling transformation. The results are validated through the image generation quality.
- **Usage**: Ensure all dependencies are installed, including PyTorch and image preprocessing libraries. Run the notebook to train the model on CelebA and evaluate its performance using metrics such as log-likelihood and image reconstructions.


### 2. Proof of Concept

- **File**: `real_nvp_sin.ipynb`
- **Description**: This notebook serves as a proof of concept for NICE by applying it to a synthetic sinusoidal 2D distribution. This experiment helps in understanding the model’s behavior and performance on simple, generated data.
- **Usage**: Run this notebook to see how Real NVP models the sinusoidal distribution and evaluates its ability to handle synthetic, non-trivial data.

## Results

- **CelebA Experiment**: The model demonstrates the ability to generate good-quality images from the latent space and achieves to get images close to those reported in the Real NVP paper. Generated images are visually inspected for quality.
- **Proof of Concept**: The notebook includes visualizations and metrics that illustrate how the model captures the structure of the sinusoidal distribution.

# Dev's note

I found the paper introducing real NVP very unclear on the exact training details. In practice, I encountered many difficulties related to numerical stability. Some of the information is not given in the paper and is a hypothesis for obtaining similar results. 

- The problem seems to stem from the fact that the deeper the normalizing flow, the more the forward pass results in latent variables with very large values: $z = f(x)$, which pushes the model to explore extreme scaling factors. An intermediate logarithmic transformation could probably solve this problem - as suggested by this [article](https://arxiv.org/html/2402.16408v1#S2.E4). I use the SoftLogCouplingLayer ($\tau = 100$) which is the function :
$$SoftLog(x) = \begin{cases}
  x, \text{if} \left|x\right| < \tau\\
  [log(1+\left|x\right| - \tau) + \tau].\texttt{sign}(x), \text{if} \left|x\right| \ge \tau  
\end{cases}$$
$$z = SoftLog(f(x))$$
In practice I don't know if we really go in the non-identity part of the function because I did not take the time to verify the $z$ values.

- I had introduced a stupid error in my code that didn't generate an error message, but the model still didn't learn. Loss decreased linearly, as if unconstrained by the pdf of the latent distribution. We also observed that the model wanted to learn the largest possible scaling factors - which is counter-intuitive with my previous comment. The problem was that I was using the log det jacobian returned by my forward pass $z, \texttt{ldj} = f(x)$ when it's really the log probability that we want to optimize: this explains why the final value of $z$ had no influence on the loss and why the scaling factors were exploding.