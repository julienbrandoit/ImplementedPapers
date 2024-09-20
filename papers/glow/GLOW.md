# GLOW Implementation

This folder contains the implementation of the GLOW model. It includes two Jupyter notebooks that demonstrate the model’s application.

## Overview

The GLOW model, as introduced in the paper *Glow: Generative Flow with Invertible 1×1 Convolutions*, is implemented here with two main experiments:

1. **CIFAR-10 Experiment**: Reproduces the results of GLOW on the CIFAR-10 dataset without any conditioning. 
 
2. **Class conditional CIFAR-10 Experiment**: Reproduce the results of GLOW on the CIFAR-10 dataset with a conditioning on the classes.

## Notebooks

### 1. CIFAR-10 Experiment

- **File**: `glow_cifar.ipynb`
- **Description**: This notebook implements the GLOW model on the CIFAR-10 dataset. It attempts to closely follow the experimental setup described in the original article, implementing specific techniques such as invertible convolution and actnorm.


### 2. Class conditional CIFAR-10 Experiment

- **File**: `glow_conditional_cifar.ipynb`
- **Description**: This notebook implements the GLOW model on the CIFAR-10 dataset with a conditional prior on the class. It attempts to closely follow the experimental setup described in the original article, although the article is rather vague about this experiment.

## Results

Both experiences can be considered successful in my opinion. In practice, the quality achieved is not amazing, but the concept is clearly promising and proven.

# Dev's note

The implementation was quite straightforward by building on top of the real NVP framework.

- The PLU decomposition is crucial for speeding up the training process. The computation of the log determinant (W) is costly.

- I encountered some issues with the inversion of the W matrix: since the training only involves the forward pass in the flow, I don’t see W becoming singular or near-singular. This leads me to consider numerical stability when inverting and in the PLU:
    1. P is straightforward.
    2. L is straightforward since we enforce the diagonal to be 1.
    3. U is problematic with the diagonal being $s$. I use $( s + \epsilon \cdot \text{sign}(s) )$ to enforce a non-zero diagonal.

- I use the same SoftLog as in my real NVP implementation. Once again, I’m not sure if we enter the non-linear region (( $\tau = 100$ )).

-  For now, the scaling is kept between -1 and 1.

- It was not very clear if they also model the logit distribution as in real NVP because they only mention using the same preprocessing (and I’m not sure if this counts as preprocessing). In practice, I do use this logit transform because it incorporates a prior in the model (we know that pixel values are between 0 and 1).

-  When performing the PLU decomposition and learning, we should enforce the structure of L and U to avoid learning outside of the triangular parts!

- The approach I used for the conditional normalizing flow, while considering the framework, is not compatible with the multi-scale architecture that overwrites the forward and inverse functions. This results in awkward code where I pass c as an input to both the forward and backward functions, which is unnecessary with only the conditional latent. I need to think about how to create a base class for a multi-scale model.

- In the conditional model, we do not restrict the mean value. Maybe adding a regularization term could improve the validation loss?

- Upon reviewing the code, I realized a significant mistake: I use .eval() for the normalizing flow model but not for the mean network. While this isn’t a problem with my simple conditional mean network for the latent distribution, it will be crucial if the network becomes more complex, especially if we introduce dropout or other components that behave differently in training and evaluation modes.
