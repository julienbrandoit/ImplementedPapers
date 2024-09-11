import torch
from .base import BaseChunker

class HalfChunker(BaseChunker):
    """
    Splits the input tensor into two halves along the feature dimension.
    
    Attributes:
        permute (bool): Whether to permute the halves after splitting.
    """
    
    def __init__(self, permute=True):
        """
        Args:
            permute (bool, optional): If True, switch the order of the halves.
        """
        super(HalfChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Split the input tensor into two halves along the feature dimension.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Two tensors split along the feature dimension.
        """
        x1, x2 = x.chunk(2, dim=1)
        return (x2, x1) if self.permute else (x1, x2)
    
    def invert(self, y1, y2):
        """
        Combine two tensors by concatenating along the feature dimension.
        
        Args:
            y1 (Tensor): First tensor.
            y2 (Tensor): Second tensor.

        Returns:
            Tensor: Combined tensor.
        """
        return torch.cat((y2, y1), dim=1) if self.permute else torch.cat((y1, y2), dim=1)

class OddEvenChunker(BaseChunker):
    """
    Splits the input tensor into odd and even indexed features.
    
    Attributes:
        permute (bool): Whether to permute the odd and even indexed features.
    """
    
    def __init__(self, permute=True):
        """
        Args:
            permute (bool, optional): If True, switch the order of odd and even features.
        """
        super(OddEvenChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Split the input tensor into odd and even indexed features.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Odd and even indexed tensors.
        """
        x_odd = x[:, 1::2]
        x_even = x[:, 0::2]
        return (x_odd, x_even) if self.permute else (x_even, x_odd)
    
    def invert(self, y1, y2):
        """
        Interleave two tensors (odd and even features) to form the original input.
        
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
            y_combined[:, 0::2] = y1  # Even positions
            y_combined[:, 1::2] = y2  # Odd positions

        return y_combined

class SpatialCheckerboardChunker(BaseChunker):
    """
    Splits the input tensor into two chunks using a spatial checkerboard pattern.
    
    Attributes:
        permute (bool): If True, reverses the order of the chunks after splitting.
    """
    
    def __init__(self, permute=True):
        """
        Args:
            permute (bool, optional): Whether to reverse the order of the chunks.
        """
        super(SpatialCheckerboardChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Splits the input tensor into two tensors based on a checkerboard pattern.
        
        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Two tensors split based on the checkerboard pattern.
        """
        x1 = torch.zeros_like(x)
        x2 = torch.zeros_like(x)
        
        # Checkerboard pattern extraction
        x1[:, :, ::2, 1::2] = x[:, :, ::2, 1::2]
        x1[:, :, 1::2, ::2] = x[:, :, 1::2, ::2]
        x2[:, :, ::2, ::2] = x[:, :, ::2, ::2]
        x2[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]

        return (x1, x2) if self.permute else (x2, x1)
    
    def invert(self, y1, y2):
        """
        Recombines two tensors using a checkerboard pattern.
        
        Args:
            y1 (Tensor): First tensor.
            y2 (Tensor): Second tensor.

        Returns:
            Tensor: Combined tensor.
        """
        x = torch.zeros_like(y1)
        
        if self.permute:
            x[:, :, ::2, 1::2] = y1[:, :, ::2, 1::2]
            x[:, :, 1::2, ::2] = y1[:, :, 1::2, ::2]
            x[:, :, ::2, ::2] = y2[:, :, ::2, ::2]
            x[:, :, 1::2, 1::2] = y2[:, :, 1::2, 1::2]
        else:
            x[:, :, ::2, 1::2] = y2[:, :, ::2, 1::2]
            x[:, :, 1::2, ::2] = y2[:, :, 1::2, ::2]
            x[:, :, ::2, ::2] = y1[:, :, ::2, ::2]
            x[:, :, 1::2, 1::2] = y1[:, :, 1::2, 1::2]
        
        return x


class ChannelWiseChunker(BaseChunker):
    """
    Splits the input tensor into two halves along the channel dimension.
    
    Attributes:
        permute (bool): If True, reverses the order of the chunks after splitting.
    """
    
    def __init__(self, permute=False):
        """
        Args:
            permute (bool, optional): Whether to reverse the order of the chunks.
        """
        super(ChannelWiseChunker, self).__init__()
        self.permute = permute
    
    def forward(self, x):
        """
        Splits the input tensor into two halves along the channel dimension.
        
        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Two tensors split along the channel dimension.
        """
        x1, x2 = x.chunk(2, dim=1)
        return (x2, x1) if self.permute else (x1, x2)
    
    def invert(self, y1, y2):
        """
        Recombines two tensors by concatenating along the channel dimension.
        
        Args:
            y1 (Tensor): First tensor.
            y2 (Tensor): Second tensor.

        Returns:
            Tensor: Combined tensor.
        """
        return torch.cat([y2, y1], dim=1) if self.permute else torch.cat([y1, y2], dim=1)
