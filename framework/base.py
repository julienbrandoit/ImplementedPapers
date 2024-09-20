import torch.nn as nn
import abc

class BaseCouplingLayer(nn.Module, abc.ABC):
    """
    Abstract base class for a coupling layer in normalizing flows.
    """
    
    def __init__(self):
        super(BaseCouplingLayer, self).__init__()
    
    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the coupling layer.

        Args:
            x (Tensor): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Transformed tensor.
        """
        pass

    @abc.abstractmethod
    def inverse(self, y, *args, **kwargs):
        """
        Inverse pass of the coupling layer.

        Args:
            y (Tensor): Input tensor in the transformed space.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Transformed tensor back to the original space.
            Scalar: Log determinant of the Jacobian.
        """
        pass

    @abc.abstractmethod
    def log_det_jacobian(self, x, *args, **kwargs):
        """
        Compute the log determinant of the Jacobian of the transformation.

        Args:
            x (Tensor): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Log determinant of the Jacobian.
        """
        pass


class BaseChunker(nn.Module, abc.ABC):
    """
    Abstract base class for splitting and combining input tensors.
    """
    
    @abc.abstractmethod
    def forward(self, x):
        """
        Split the input tensor into chunks.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Two chunks of the input tensor.
        """
        pass
    
    @abc.abstractmethod
    def invert(self, y1, y2):
        """
        Combine two chunks into one tensor.

        Args:
            y1 (Tensor): First chunk.
            y2 (Tensor): Second chunk.

        Returns:
            Tensor: Combined tensor.
        """
        pass
