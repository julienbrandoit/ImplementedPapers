"""
# My Deep Learning Framework

This package contains implementations of various components for deep learning models, including 
coupling layers, chunkers, and normalizing flows. It is designed to be modular and extendable, 
allowing easy integration of new layers and models.

## Modules
- **base**: Base classes and utilities
- **coupling_layers**: Implementations of specific coupling layers
- **chunkers**: Implementations of chunkers
- **normalizing_flows**: Definition of normalizing flow models
- **distributions**: Custom distribution implementations (e.g., Logistic distribution)

## Note on Implementation

While I reimplement papers from scratch by hand to ensure a deep understanding of the concepts,
I also use tools such as ChatGPT and GitHub Copilot to assist with generating code.
These tools help ensure coherent naming conventions and function descriptions, facilitating a more organized and consistent framework development process.
"""

from .base import BaseCouplingLayer, BaseChunker
from .coupling_layers import AdditiveCouplingLayer, ScalingCouplingLayer
from .chunkers import HalfChunker, OddEvenChunker
from .normalizing_flows import NormalizingFlow
from .distributions import Logistic

__all__ = [
    'BaseCouplingLayer',
    'BaseChunker',
    'AdditiveCouplingLayer',
    'ScalingCouplingLayer',
    'HalfChunker',
    'OddEvenChunker',
    'NormalizingFlow',
    'Logistic'
]
