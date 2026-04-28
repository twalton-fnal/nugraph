"""Position features transform"""
import torch
from torch import cat
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from pynuml.data import NuGraphData

class PositionFeatures(BaseTransform):
    """
    Append node position tensor to node feature tensor
    
    Args:
        planes: list of detector plane names
    """
    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Apply a transform to concatenate the node position onto the node feature tensor

        Args:
           data: NuGraph data object to transform
        """
        
        # in second-generation inputs, hit nodes live in a single node store
        if "hit" in data.node_types:
            node_types = ("hit",)

        # in first-generation inputs, hit nodes are separated out by plane
        else:
            node_types = self.planes

        for node_type in node_types:
            n = data[node_type]
            n.x = torch.cat((n.pos, n.x), dim=-1)
        
        return data
        