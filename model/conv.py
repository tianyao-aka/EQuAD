import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConv(MessagePassing):

    def __init__(self, in_dim,emb_dim,edge_dim=-1):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
        #                                torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim),torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU())
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        if edge_dim>1:
            self.edge_encoder = torch.nn.Linear(edge_dim, in_dim)
        self.edge_dim = edge_dim


    def forward(self, x, edge_index, edge_attr=None,edge_weight=None):

        if self.edge_dim > 0 and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x,edge_attr=edge_attr, edge_weight=edge_weight))
        return out

        
    def message(self, x_j,edge_attr, edge_weight):
        if edge_attr is not None and self.edge_dim>0:
            # print (x_j.shape,edge_attr.shape)
            x_j = x_j + edge_attr
            
        if edge_weight is not None:
            return F.relu(x_j * edge_weight.view(-1,1))
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out
    
    