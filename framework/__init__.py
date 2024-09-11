"""
# My Deep Learning Framework

This package contains implementations of various components for deep learning models, including 
coupling layers, chunkers, and normalizing flows. It is designed to be modular and extendable, 
allowing easy integration of new layers and models.

## Modules
- **base**: Base classes and utilities for defining coupling layers and chunkers.
  - `BaseCouplingLayer`: Abstract base class for coupling layers.
  - `BaseChunker`: Abstract base class for chunking operations.

- **coupling_layers**: Implementations of specific coupling layers.
  - `AdditiveCouplingLayer`: A coupling layer where one part of the input is added to a function of another part.
  - `ScalingCouplingLayer`: A coupling layer that scales one part of the input by a learned factor.
  - `AffineCouplingLayer`: A coupling layer applying an affine transformation.
  - `LogitCouplingLayer`: A coupling layer using a logit transformation to map the input.
  - `SoftLogCouplingLayer`: A coupling layer applying a soft logarithmic transformation.

- **chunkers**: Implementations of chunkers for splitting and combining tensors.
  - `HalfChunker`: Splits tensors into halves along the feature dimension.
  - `OddEvenChunker`: Splits tensors into odd and even indexed features.
  - `SpatialCheckerboardChunker`: Splits tensors based on a spatial checkerboard pattern.
  - `ChannelWiseChunker`: Splits tensors into halves along the channel dimension.

- **normalizing_flows**: Definition of normalizing flow models.
  - `NormalizingFlow`: A normalizing flow model composed of a stack of coupling layers, with methods for forward, inverse, and sampling operations.

- **distributions**: Custom distribution implementations.
  - `Logistic`: A custom logistic distribution class for use with normalizing flows.

## Note on Implementation

While I reimplement papers from scratch by hand to ensure a deep understanding of the concepts,
I also use tools such as ChatGPT and GitHub Copilot to assist with generating documentations.
These tools help ensure coherent naming conventions and function descriptions, facilitating a more organized and consistent framework development process.

## Available Classes and Functions

### Base Classes
- **BaseCouplingLayer**: Abstract class for coupling layers, requiring implementations of `forward`, `inverse`, and `log_det_jacobian` methods.
- **BaseChunker**: Abstract class for chunking operations, requiring implementations of `forward` and `invert` methods.

### Coupling Layers
- **AdditiveCouplingLayer**: Implements additive coupling transformations.
- **ScalingCouplingLayer**: Implements scaling transformations.
- **AffineCouplingLayer**: Implements affine transformations.
- **LogitCouplingLayer**: Implements a logit transformation for coupling.
- **SoftLogCouplingLayer**: Implements a soft logarithmic transformation.
- **BatchNormCouplingLayer**: Implements batch normalization for coupling.

### Chunkers
- **HalfChunker**: Splits tensors into two halves.
- **OddEvenChunker**: Splits tensors into odd and even indexed features.
- **SpatialCheckerboardChunker**: Splits tensors based on a spatial checkerboard pattern.
- **ChannelWiseChunker**: Splits tensors along the channel dimension.

### Normalizing Flows
- **NormalizingFlow**: Defines a normalizing flow model with methods for forward propagation, inversion, and sampling.

### Distributions
- **Logistic**: Custom logistic distribution implementation for use in normalizing flows.

"""
from .base import BaseCouplingLayer, BaseChunker
from .coupling_layers import AdditiveCouplingLayer, ScalingCouplingLayer, AffineCouplingLayer, LogitCouplingLayer, SoftLogCouplingLayer, BatchNormCouplingLayer
from .chunkers import HalfChunker, OddEvenChunker, SpatialCheckerboardChunker, ChannelWiseChunker
from .normalizing_flows import NormalizingFlow
from .distributions import Logistic

__all__ = [
    'BaseCouplingLayer',
    'BaseChunker',
    'AdditiveCouplingLayer',
    'ScalingCouplingLayer',
    'AffineCouplingLayer',
    'LogitCouplingLayer',
    'SoftLogCouplingLayer',
    'BatchNormCouplingLayer',
    'HalfChunker',
    'OddEvenChunker',
    'SpatialCheckerboardChunker',
    'ChannelWiseChunker',
    'NormalizingFlow',
    'Logistic'
]
