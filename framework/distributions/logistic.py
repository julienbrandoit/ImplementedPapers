import torch
import torch.distributions as dist

class Logistic(torch.distributions.TransformedDistribution):
    def __init__(self, loc, scale, device='cpu'):
        """
        Initialize a Logistic distribution with a given location and scale.

        Args:
            loc (float or torch.Tensor): Location parameter of the distribution.
            scale (float or torch.Tensor): Scale parameter of the distribution.
            device (str, optional): Device to place tensors on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.loc = torch.tensor(loc, device=device)
        self.scale = torch.tensor(scale, device=device)
        
        base_distribution = dist.Uniform(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        
        transforms = [
            dist.SigmoidTransform().inv,
            dist.AffineTransform(loc=self.loc, scale=self.scale)  
        ]
        
        super().__init__(base_distribution, transforms)

    def __repr__(self):
        return f"Logistic(loc={self.loc}, scale={self.scale}, device={self.loc.device})"
