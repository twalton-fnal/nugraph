import torch
from torch.autograd import Function


class ReverseLayerF(Function):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Neural Networks (DANN).

    This layer acts as the identity function in the forward pass but multiplies
    the gradients by -alpha during the backward pass. It enables adversarial
    training for domain adaptation by encouraging the feature extractor to learn
    domain-invariant representations.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor passed through unchanged in the forward pass.
    alpha : float
        Scaling factor for gradient reversal. Larger values increase the strength
        of domain adversarial learning.

    Returns
    -------
    torch.Tensor
        Same as the input tensor during forward pass.

    Notes
    -----
    - Forward: identity operation (returns `x` as is).
    - Backward: reverses gradients by multiplying with `-alpha`.
    - This mechanism is central to DANN, where a domain classifier tries to
      distinguish source and target domains while the feature extractor
      learns to fool it, thereby reducing domain shift.
    """
    
    def __init__(self):
        super().__init__()

    #@staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    #@staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
