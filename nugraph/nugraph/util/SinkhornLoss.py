import torch
from functools import partial
from torch import Tensor
import geomloss


class Sinkhorn(torch.nn.Module):
    """
    Differentiable Sinkhorn Divergence loss for comparing probability 
    distributions, following the SIDDA paper (https://arxiv.org/abs/2501.14048) 
    and implemented via GeomLoss (https://www.kernel-operations.io/geomloss/).

    This class computes the Sinkhorn divergence between two sets of samples 
    (e.g., feature embeddings from source and target domains) and is useful 
    for domain adaptation, distribution alignment, and generative modeling.

    Parameters
    ----------
    blur : float
        Regularization parameter controlling the entropy of the optimal 
        transport problem. Larger values increase smoothness at the cost 
        of precision, while smaller values give sharper transport plans.

    Methods
    -------
    forward(x, y, blur=None):
        Compute the Sinkhorn divergence between samples x and y.
        
        Parameters
        ----------
        x : torch.Tensor, shape (n_samples_x, d)
            Sample embeddings from the first distribution.
        y : torch.Tensor, shape (n_samples_y, d)
            Sample embeddings from the second distribution.
        blur : float, optional
            Overrides the stored default GeomLoss blur parameter if provided.
        
        Returns
        -------
        torch.Tensor
            Scalar Sinkhorn divergence value between x and y.
    """
    
    def __init__(self, blur):
        super().__init__()
        self.blur = blur  # Store blur in self

    def forward(self, x, y, blur=None):
        if blur is None:
            blur = self.blur  # Use the default blur if not provided

        # Create a new SamplesLoss instance with the correct blur each time
        loss = geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)
        return loss(x, y) 


# Utility functions for divergence measures
# ------------------------------------------
# - kl_divergence(p, q): Kullback–Leibler divergence
# - jensen_shannon_divergence(p, q): Symmetrized and smoothed KL
# - jensen_shannon_distance(p, q): Metric version of JSD (square root)
#
# These can be used as alternative measures of distributional shift, 
# complementing the Sinkhorn divergence.

@staticmethod
def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)

@staticmethod
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

@staticmethod
def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


