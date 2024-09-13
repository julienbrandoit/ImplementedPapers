import torch
import torch.nn as nn
from .base import BaseCouplingLayer

class AdditiveCouplingLayer(BaseCouplingLayer):
    """
    Additive coupling layer for normalizing flows.
    
    Attributes:
        coupling_function (nn.Module): Function applied to one part of the input.
        chunker (object): Utility to split and recombine the input tensor.
    """
    
    def __init__(self, coupling_function, chunker):
        """
        Args:
            coupling_function (nn.Module): Function to apply to a chunk of the input.
            chunker (object): Utility for splitting the input tensor into two parts.
        """
        super(AdditiveCouplingLayer, self).__init__()
        self.coupling_function = coupling_function
        self.chunker = chunker

    def forward(self, x):
        """
        Forward pass of the additive coupling layer.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and log determinant (zeros).
        """
        x1, x2 = self.chunker(x)
        y1 = x1
        y2 = x2 + self.coupling_function(x1)
        return self.chunker.invert(y1, y2), torch.zeros(x.size(0), device=x.device)
    
    def inverse(self, y):
        """
        Inverse pass of the additive coupling layer.
        
        Args:
            y (Tensor): Transformed tensor.
        
        Returns:
            Tensor: Original tensor before transformation.
        """
        y1, y2 = self.chunker(y)
        x1 = y1
        x2 = y2 - self.coupling_function(x1)
        return self.chunker.invert(x1, x2)
    
    def log_det_jacobian(self, x):
        """
        Log determinant of the Jacobian (zeros for additive coupling).
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Zeros tensor for the log determinant.
        """
        return torch.zeros(x.size(0), device=x.device)

class ScalingCouplingLayer(BaseCouplingLayer):
    """
    Scaling coupling layer with learnable scaling factors.
    
    Attributes:
        dim (int): Input dimensionality.
        log_scaling_factors (nn.Parameter): Log scaling factors.
    """
    
    def __init__(self, dim):
        """
        Args:
            dim (int): Input dimensionality.
        """
        super(ScalingCouplingLayer, self).__init__()
        self.dim = dim
        self.log_scaling_factors = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass of the scaling layer.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tuple[Tensor, Tensor]: Scaled input and log determinant of Jacobian.
        """
        scale = torch.exp(self.log_scaling_factors)
        return x * scale, self.log_scaling_factors.sum(dim=0).repeat(x.size(0))
    
    def inverse(self, y):
        """
        Inverse pass of the scaling layer.
        
        Args:
            y (Tensor): Transformed tensor.
        
        Returns:
            Tensor: Original tensor before scaling.
        """
        scale = torch.exp(-self.log_scaling_factors)
        return y * scale
    
    def log_det_jacobian(self, x):
        """
        Log determinant of the Jacobian for the scaling layer.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Log determinant.
        """
        return self.log_scaling_factors.sum(dim=0).repeat(x.size(0))

class SqueezingCouplingLayer(BaseCouplingLayer):
    """
    A coupling layer that applies a squeezing operation to rearrange spatial dimensions into the channel dimension.
    
    Attributes:
        factor (int): Factor by which the spatial dimensions are squeezed into the channel dimension.
    """
    
    def __init__(self, factor=2):
        """
        Args:
            factor (int, optional): The factor to squeeze the spatial dimensions by. Defaults to 2.
        """
        super(SqueezingCouplingLayer, self).__init__()
        self.factor = factor
    
    def forward(self, x):
        """
        Squeezes the input tensor by reducing its spatial dimensions and increasing the channel dimension.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tuple[Tensor, Tensor]: Squeezed tensor and log determinant (zero in this case).
        """
        b, c, h, w = x.size()
        x = x.view(b, c, h//self.factor, self.factor, w//self.factor, self.factor) \
             .permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c*self.factor*self.factor, h//self.factor, w//self.factor), torch.zeros(x.size(0), device=x.device)

    def inverse(self, y):
        """
        Applies the inverse squeezing operation to expand the spatial dimensions back from the channel dimension.

        Args:
            y (Tensor): Squeezed input tensor.

        Returns:
            Tensor: Re-expanded tensor with original spatial dimensions.
        """
        return self.invert(y)
    
    def invert(self, y):
        """
        Inverts the squeezing operation.

        Args:
            y (Tensor): Squeezed input tensor.

        Returns:
            Tensor: Re-expanded tensor.
        """
        b, c, h, w = y.size()
        y = y.view(b, c//(self.factor*self.factor), self.factor, self.factor, h, w) \
             .permute(0, 1, 4, 2, 5, 3).contiguous()
        return y.view(b, c//(self.factor*self.factor), h*self.factor, w*self.factor)
    
    def log_det_jacobian(self, x):
        """
        Returns zero as the log determinant of the Jacobian for this transformation is zero.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Zero tensor.
        """
        return torch.zeros(x.size(0), device=x.device)


class SequentialCouplingLayer(BaseCouplingLayer):
    """
    A coupling layer that applies a sequence of other coupling layers in order.
    
    Attributes:
        layers (ModuleList): List of coupling layers to be applied sequentially.
    """
    
    def __init__(self, layers):
        """
        Args:
            layers (list): List of coupling layers to apply sequentially.
        """
        super(SequentialCouplingLayer, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """
        Passes the input through each coupling layer in sequence, accumulating the log determinants of the Jacobians.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and accumulated log determinant.
        """
        log_det_jac = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_jac += ldj
        return x, log_det_jac
    
    def inverse(self, y):
        """
        Passes the input through each coupling layer in reverse order to invert the transformation.

        Args:
            y (Tensor): Input tensor in the transformed space.

        Returns:
            Tensor: Reconstructed tensor in the original space.
        """
        x = y
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
    
    def log_det_jacobian(self, x):
        """
        Computes the total log determinant of the Jacobians for the entire sequence of coupling layers.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Accumulated log determinant.
        """
        _, ldj = self(x)
        return ldj
    
class LogitCouplingLayer(BaseCouplingLayer):
    """
    A coupling layer that applies a logit transformation to the input tensor.
    
    Attributes:
        alpha (float): The minimum value to avoid logit saturation.
        logit (function): Logit transformation function.
        lalpha (Tensor): Precomputed log value for alpha to use in the log determinant calculation.
    """
    
    def __init__(self, alpha=0.05):
        """
        Args:
            alpha (float, optional): Minimum value to avoid logit saturation. Defaults to 0.05.
        """
        super(LogitCouplingLayer, self).__init__()
        self.alpha = alpha
        self.logit = torch.logit
        self.lalpha = torch.log(torch.tensor(1-alpha))
    
    def forward(self, x):
        """
        Applies the logit transformation to the input tensor and computes the log determinant of the Jacobian.
        
        Transformation:
            z = logit(alpha + (1 - alpha) * x)
        
        Log determinant of the Jacobian:
            ldj = -sum(log(xp * (1 - xp))) + (num_elements / batch_size) * lalpha
        
        Args:
            x (Tensor): Input tensor with values in the range [0, 1].

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and log determinant of the Jacobian.
        """
        xp = self.alpha + (1 - self.alpha) * x
        z = self.logit(xp)
        
        lf = x.numel() / x.size(0)
        
        ldj = torch.sum(-torch.log(xp * (1 - xp)).view(x.size(0), -1), dim=1) + lf * self.lalpha
        return z, ldj
    
    def inverse(self, z):
        """
        Applies the inverse logit transformation to recover the original tensor.
        
        Inverse transformation:
            x = (sigmoid(z) - alpha) / (1 - alpha)
        
        Args:
            z (Tensor): Input tensor in the logit space.

        Returns:
            Tensor: Reconstructed tensor in the original space.
        """
        xp = torch.sigmoid(z)
        x = (xp - self.alpha) / (1 - self.alpha)
        return x
    
    def log_det_jacobian(self, x):
        """
        Computes the log determinant of the Jacobian for the logit transformation.
        
        Log determinant of the Jacobian:
            ldj = -sum(log(xp * (1 - xp))) + (num_elements / batch_size) * lalpha
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log determinant of the Jacobian.
        """
        xp = self.alpha + (1 - self.alpha) * x
        lf = x.numel() / x.size(0)
        ldj = torch.sum(-torch.log(xp * (1 - xp)).view(x.size(0), -1), dim=1) + lf * self.lalpha
        return ldj

class SoftLogCouplingLayer(BaseCouplingLayer):
    """
    A coupling layer that applies a soft logarithmic transformation to the input tensor.
    
    Attributes:
        tau (float): Threshold parameter for the soft logarithmic transformation.
    """
    
    def __init__(self, tau=100):
        """
        Args:
            tau (float, optional): Threshold parameter. Defaults to 100.
        """
        super(SoftLogCouplingLayer, self).__init__()
        self.tau = tau
    
    def forward(self, x):
        """
        Applies the soft logarithmic transformation to the input tensor and computes the log determinant of the Jacobian.
        
        Transformation:
            For |x| >= tau:
                uz = log1p(|x| - tau) + tau
            For |x| < tau:
                uz = |x|
            
            z = uz * sign(x)
        
        Log determinant of the Jacobian:
            For |x| >= tau:
                ldj_uz = log1p(|x| - tau)
            For |x| < tau:
                ldj_uz = 0
            log_det_jacobian = -sum(ldj_uz)
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Transformed tensor and log determinant of the Jacobian.
        """
        abs_x = torch.abs(x)
        uz = torch.where(abs_x >= self.tau,
                         torch.log1p(abs_x - self.tau) + self.tau,
                         abs_x)
        
        ldj_uz = torch.where(abs_x >= self.tau,
                             torch.log1p(abs_x - self.tau),
                             torch.tensor(0.0, device=x.device))
        z = uz * torch.sign(x)
        log_det_jacobian = -torch.sum(ldj_uz.view(x.size(0), -1), dim=1)
        
        return z, log_det_jacobian
    
    def inverse(self, z):
        """
        Applies the inverse soft logarithmic transformation to recover the original tensor.
        
        Inverse transformation:
            For |z| >= tau:
                x = expm1(|z| - tau) + tau
            For |z| < tau:
                x = |z|
            
        Args:
            z (Tensor): Input tensor in the transformed space.

        Returns:
            Tensor: Reconstructed tensor in the original space.
        """
        abs_z = torch.abs(z)
        x = torch.where(abs_z >= self.tau,
                        torch.expm1(abs_z - self.tau) + self.tau,
                        abs_z) * torch.sign(z)
        return x
    
    def log_det_jacobian(self, x):
        """
        Computes the log determinant of the Jacobian for the soft logarithmic transformation.
        
        Log determinant of the Jacobian:
            For |x| >= tau:
                ldj_uz = log1p(|x| - tau)
            For |x| < tau:
                ldj_uz = 0
            log_det_jacobian = -sum(ldj_uz)
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log determinant of the Jacobian.
        """
        abs_x = torch.abs(x)
        ldj_uz = torch.where(abs_x >= self.tau,
                             torch.log1p(abs_x - self.tau),
                             torch.tensor(0.0, device=x.device))
        return -torch.sum(ldj_uz.view(x.size(0), -1), dim=1)

class BatchNormCouplingLayer(BaseCouplingLayer):
    """
    Batch normalization coupling layer. Normalizes input using batch statistics during training and running averages during evaluation.
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(BatchNormCouplingLayer, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.register_buffer('avg_mean', torch.zeros(num_features))
        self.register_buffer('avg_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            with torch.no_grad():
                self.avg_mean = self.momentum * self.avg_mean + (1 - self.momentum) * batch_mean
                self.avg_var = self.momentum * self.avg_var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.avg_mean
            batch_var = self.avg_var
        
        mean = batch_mean.view(1, -1, 1, 1)
        var = batch_var.view(1, -1, 1, 1)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        log_det = -0.5 * torch.sum(torch.log(var + self.eps), dim=(1, 2, 3)) + 0.5 * torch.log(self.eps) * x.numel() / x.size(1)
        
        return normalized, log_det

    def inverse(self, x):
        mean = self.avg_mean.view(1, -1, 1, 1)
        var = self.avg_var.view(1, -1, 1, 1)
        return x * torch.sqrt(var + self.eps) + mean
    
    def log_det_jacobian(self, x):
        var = self.avg_var.view(1, -1, 1, 1)
        return -0.5 * torch.sum(torch.log(var + self.eps), dim=(1, 2, 3)) + 0.5 * torch.log(self.eps) * x.numel() / x.size(1)

class AffineCouplingLayer(BaseCouplingLayer):
    def __init__(self, coupling_function, chunker, should_mask = True):
        super(AffineCouplingLayer, self).__init__()
        self.coupling_function = coupling_function
        self.chunker = chunker
        self.should_mask = should_mask

    def forward(self, x):
        x1, x2 = self.chunker(x)

        scale, shift = self.coupling_function(x1).chunk(2, dim=1)
        y1 = x1
        y2 = x2*torch.exp(scale) + shift
        
        if self.should_mask:
            _, masked_scale = self.chunker(scale)
        else :
            masked_scale = scale
        return self.chunker.invert(y1, y2), masked_scale.view(x.size(0), -1).sum(dim=1)
    
    def inverse(self, y):
        y1, y2 = self.chunker(y)
        scale, shift = self.coupling_function(y1).chunk(2, dim=1)
        x1 = y1
        x2 = (y2 - shift)*torch.exp(-scale)
        return self.chunker.invert(x1, x2)
    
    def log_det_jacobian(self, x):
        x1, x2 = self.chunker(x)
        scale, _ = self.coupling_function(x1).chunk(2, dim=1)
        if self.should_mask:
            _, masked_scale = self.chunker(scale)
        else :
            masked_scale = scale
        return masked_scale.view(x.size(0), -1).sum(dim=1)
