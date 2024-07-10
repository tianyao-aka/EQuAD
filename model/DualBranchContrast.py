import torch

from GCL.losses import Loss
from GCL.models import get_sampler


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask



def generate_mask(num_graphs, num_nodes, ptr, node_indices, node_masks, node_indice_len):
    """
    Generates a mask tensor M of shape (num_graphs, num_nodes) with specified modifications.
    Args:
    - num_graphs (int): Number of graphs in the batch.
    - num_nodes (int): Total number of nodes.
    - ptr (torch.Tensor): Indices for each graph.
    - node_indices (torch.Tensor): 1D tensor of node indices.
    - node_masks (torch.Tensor): Mask for the node_indices.
    - node_indice_len (torch.Tensor): Tensor indicating the size for each graph in node_indices.
    """

    # Initialize the mask tensor M with zeros
    M = torch.zeros((num_graphs, num_nodes), dtype=torch.bool)

    # Iterate over each graph
    index_pointer = 0
    for i in range(num_graphs):
        # Get the start and end index for the current graph
        start_idx = ptr[i]
        end_idx = ptr[i + 1]

        # Set the mask for all nodes in the current graph to 1
        M[i, start_idx:end_idx] = 1

        # Get the node indices and masks for the current graph
        current_node_indices = node_indices[index_pointer:index_pointer + node_indice_len[i]].long()
        current_node_masks = node_masks[index_pointer:index_pointer + node_indice_len[i]].long()

        # Apply the node mask to set the corresponding nodes to 0 if the mask is True
        M[i, current_node_indices[current_node_masks]] = 0
        # Move the index pointer
        index_pointer += node_indice_len[i]
    pos_mask = M*1.0
    return pos_mask


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False,biased=False,device=0, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs
        self.biased = biased
        self.device = device

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None,data=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)
        
        

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5



class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5