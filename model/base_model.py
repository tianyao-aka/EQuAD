"""
use test data to calculate the loss in SRGNN
"""
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn
import torch
from torch_geometric.utils import dropout_adj


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    

    def _ensure_contiguousness(self,
                               x,
                               edge_idx,
                               edge_attr,
                               edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        if edge_attr is not None:
            edge_attr = edge_attr.contiguous()
        
        return x, edge_idx, edge_attr, edge_weight
    
