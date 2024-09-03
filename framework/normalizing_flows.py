import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    def __init__(self, layers, latent_dim, latent_distribution, forward_transform=None, inverse_transform=None, device=None):
        """
        Initialize the normalizing flow model.
        
        Args:
            layers (list of nn.Module): List of coupling layers to stack in the flow.
            latent_dim (int): Dimensionality of the latent space.
            latent_distribution (torch.distributions.Distribution): Latent space distribution.
            forward_transform (callable, optional): Function for the forward transformation of the input.
            inverse_transform (callable, optional): Function for the inverse transformation of the output.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). If None, use available device.
        """
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.latent_dim = latent_dim
        self.latent_distribution = latent_distribution
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.to(device)
        
        self.latent_distribution.loc = self.latent_distribution.loc.to(self.device)
        self.latent_distribution.scale = self.latent_distribution.scale.to(self.device)

    def forward(self, x):
        """
        Perform the forward pass of the normalizing flow.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and the log determinant of the Jacobian.
        """
        log_det_jacobian = 0
        if self.forward_transform:
            x = self.forward_transform(x)
        for layer in self.layers:
            x = layer(x)
            log_det_jacobian += layer.log_det_jacobian(x)
        return x, log_det_jacobian

    def inverse(self, y):
        """
        Perform the inverse pass of the normalizing flow.
        
        Args:
            y (Tensor): Transformed tensor.

        Returns:
            Tensor: Original tensor.
        """
        if self.inverse_transform:
            y = self.inverse_transform(y)
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def sample(self, n_samples):
        """
        Sample from the normalizing flow model by sampling from the latent distribution and transforming.
        
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Tensor: Samples from the normalizing flow.
        """
        z = self.latent_distribution.sample((n_samples, self.latent_dim))
        device = next(self.parameters()).device
        z = z.to(device)
        return self.inverse(z)

    def log_prob(self, x):
        """
        Compute the log probability of the input under the model.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log probability of the input.
        """
        z, log_det_jacobian = self.forward(x)
        return torch.sum(self.latent_distribution.log_prob(z), dim=-1) + log_det_jacobian
