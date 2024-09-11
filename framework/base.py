import torch.nn as nn
import abc

class BaseCouplingLayer(nn.Module, abc.ABC):
    def __init__(self):
        super(BaseCouplingLayer, self).__init__()
    
    @abc.abstractmethod
    def forward(self, x):
        """
        Apply the forward pass of the coupling layer.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed tensor.
        """
        pass

    @abc.abstractmethod
    def inverse(self, y):
        """
        Apply the inverse pass of the coupling layer.
        
        Args:
            y (Tensor): Input tensor in the transformed space.

        Returns:
            Tensor: Transformed tensor back to the original space.
            Scalar: Log determinant of the Jacobian matrix.
        """
        pass

    @abc.abstractmethod
    def log_det_jacobian(self, x):
        """
        Compute the log determinant of the Jacobian matrix of the transformation.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log determinant of the Jacobian matrix.
        """
        pass


class BaseChunker(nn.Module, abc.ABC):
    
    @abc.abstractmethod
    def forward(self, x):
        """
        Split the input tensor into chunks.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of chunks.
        """
        pass
    
    @abc.abstractmethod
    def invert(self, y1, y2):
        """
        Combine two tensors into one.

        Args:
            y1 (Tensor): First chunk.
            y2 (Tensor): Second chunk.

        Returns:
            Tensor: Combined tensor.
        """
        pass
