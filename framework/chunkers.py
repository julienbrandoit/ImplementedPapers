import torch
from .base import BaseChunker

class HalfChunker(BaseChunker):
    def __init__(self, permute=True):
        super(HalfChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Split the input tensor into two halves along the feature dimension.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of two tensors split along the feature dimension.
        """
        x1, x2 = x.chunk(2, dim=1)
        if self.permute:
            return x2, x1
        else:
            return x1, x2
    
    def inverse(self, y):
        """
        Combine two tensors back into a single tensor by concatenating along the feature dimension.
        
        Args:
            y (Tuple[Tensor, Tensor]): Tuple of two tensors.

        Returns:
            Tensor: Combined tensor.
        """
        y1, y2 = y
        if self.permute:
            return torch.cat((y2, y1), dim=1)
        else:
            return torch.cat((y1, y2), dim=1)
    
    def invert(self, y1, y2):
        """
        Combine two tensors into one by concatenating along the feature dimension.
        
        Args:
            y1 (Tensor): First tensor.
            y2 (Tensor): Second tensor.

        Returns:
            Tensor: Combined tensor.
        """
        if self.permute:
            return torch.cat((y2, y1), dim=1)
        else:
            return torch.cat((y1, y2), dim=1)

class OddEvenChunker(BaseChunker):
    def __init__(self, permute=True):
        super(OddEvenChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Split the input tensor into odd and even indexed features.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of odd and even indexed tensors.
        """
        x_odd = x[:, 1::2]
        x_even = x[:, 0::2]
        
        if self.permute:
            return x_odd, x_even
        else:
            return x_even, x_odd
    
    def inverse(self, y):
        """
        Combine odd and even indexed tensors back into a single tensor by interleaving the features.
        
        Args:
            y (Tuple[Tensor, Tensor]): Tuple of odd and even indexed tensors.

        Returns:
            Tensor: Combined tensor.
        """
        y_odd, y_even = y
        batch_size = y_odd.size(0)
        num_features = y_even.size(1) + y_odd.size(1)
        
        y_combined = torch.empty((batch_size, num_features), device=y_even.device, dtype=y_even.dtype)
        
        if self.permute:
            y_combined[:, 0::2] = y_even  # Even positions
            y_combined[:, 1::2] = y_odd   # Odd positions
        else:
            y_combined[:, 0::2] = y_odd   # Odd positions
            y_combined[:, 1::2] = y_even  # Even positions

        return y_combined
    
    def invert(self, y1, y2):
        """
        Combine two tensors into one by interleaving the features.
        
        Args:
            y1 (Tensor): Tensor containing odd indexed features.
            y2 (Tensor): Tensor containing even indexed features.

        Returns:
            Tensor: Combined tensor.
        """
        batch_size = y1.size(0)
        num_features = y1.size(1) + y2.size(1)
        
        y_combined = torch.empty((batch_size, num_features), device=y1.device, dtype=y1.dtype)
        
        if self.permute:
            y_combined[:, 0::2] = y2  # Even positions
            y_combined[:, 1::2] = y1  # Odd positions
        else:
            y_combined[:, 0::2] = y1  # Odd positions
            y_combined[:, 1::2] = y2  # Even positions

        return y_combined
