import torch
from functools import partial
from torch import Tensor



class MMDLoss(torch.nn.Module):
    """
        Maximum Mean Discrepancy (MMD) Loss for domain adaptation.
    
        This loss measures the distance between the distributions of 
        source and target domain feature representations in a 
        reproducing kernel Hilbert space (RKHS). By minimizing the MMD, 
        the model encourages alignment between the two feature 
        distributions, promoting domain-invariant representations.
    
        The implementation uses a multi-scale Gaussian kernel with a 
        range of bandwidths (sigmas) to capture discrepancies at 
        different scales.
        """
    def __init__(self):
        super().__init__()
        self.sigmas = torch.Tensor([
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
            10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6
        ]).float().cuda()

    def forward(self, hs: Tensor, ht: Tensor) -> Tensor:
        gaussian_kernel = partial(self._gaussian_kernel_matrix, sigmas=self.sigmas)
        loss_value = self._maximum_mean_discrepancy(hs, ht, kernel=gaussian_kernel)
        return torch.clamp(loss_value, min=1e-4)

    @staticmethod
    def _compute_pairwise_distances(x: Tensor, y: Tensor) -> Tensor:
        """Computes the squared pairwise Euclidean distances between x and y.
        Args:
            x (torch.Tensor): a tensor of shape [num_x_samples, num_features]
            y (torch.Tensor): a tensor of shape [num_y_samples, num_features]
    
        Returns:
            torch.Tensor: a distance matrix of dimensions [num_x_samples, num_y_samples].
        """
        if not x.dim() == y.dim() == 2:
            raise ValueError('Both inputs should be 2D matrices.')
        if x.size(1) != y.size(1):
            raise ValueError('Inputs must have the same number of features.')

        norm = lambda x: torch.sum(torch.pow(x, 2), 1)
        return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)


    def _gaussian_kernel_matrix(self, x: Tensor, y: Tensor, sigmas: Tensor) -> Tensor:
        """Multi-scale Gaussian RBF kernel for MMD.
        Args:
            x (torch.Tensor): a tensor of shape [num_samples, num_features]
            y (torch.Tensor): a tensor of shape [num_samples, num_features]
            sigmas: free parameter that determins the width of the kernel
    
        Returns:
            torch.Tensor: a tensor of shape [num_x_samples, num_y_samples] with the kernel values
        """
        beta = 1. / (2. * torch.unsqueeze(sigmas, 1))
        dist = self._compute_pairwise_distances(x, y)
        s = torch.matmul(beta, dist.contiguous().view(1, -1))
        return torch.sum(torch.exp(-s), 0).view(*dist.size())

    def _maximum_mean_discrepancy(self, x: Tensor, y: Tensor, kernel) -> Tensor:
        """Computes the Maximum Mean Discrepancy (MMD) of two samples.
        Args:
            x (torch.Tensor): a tensor of shape [num_samples, num_features]
            y (torch.Tensor): a tensor of shape [num_samples, num_features]
            kernel (function, optional): kernel function. Defaults to gaussian_kernel_matrix.
            
        Returns:
            torch.Tensor: a scalar denoting the squared maximum mean discrepancy loss.
        """
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return torch.clamp(cost, min=0.0)