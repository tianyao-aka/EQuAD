import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
try:
    from base_model import BaseModel
    from conv import GCNConv
except:
    print ('import from root dir')
    from model.base_model import BaseModel
    from model.conv import GCNConv
    
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation



class GCN(BaseModel):
    def __init__(self, nfeat, nhid,nclass, nlayers=2, dropout=0.0,save_mem=True,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6, with_bias=True, **args):

        super(GCN, self).__init__()
        self.jk = jk
        self.node_cls = node_cls
        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()
        
        if nlayers == 1:
            self.layers.append(GCNConv(nfeat, nhid))
        else:
            self.layers.append(GCNConv(nfeat, nhid))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-1):
                self.layers.append(GCNConv(nhid, nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        if self.jk=='concat':
            self.jk_layer = nn.Linear(nhid * nlayers, nhid)
        if pooling=='sum':
            self.pool = global_add_pool
        if pooling=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(nhid,1))
        self.name = 'GCN'

    def forward(self, x, edge_index,edge_weight=None,batch=None,return_both_rep=False,edge_attr=None):
        if not self.node_cls:
            assert batch is not None, "Please specify 'batch' for graph pooling!"
        xs = []
        # x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        # if edge_weight is not None:
        #     adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                # x = layer(x, edge_index, edge_weight=edge_weight)
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
            xs.append(x)
        
        if self.jk=='last':
            if self.node_cls:
                return x
            else:
                g = self.pool(x,batch)
                return (x,g) if return_both_rep else g # return node rep then graph rep
        else:
            x = torch.cat(xs, dim=1)
            x = self.jk_layer(x)
            if self.node_cls:
                return x
            else:
                g = self.pool(x,batch)
                return (x,g) if return_both_rep else g # return node rep then graph rep
        # return F.log_softmax(x, dim=1)
    
    
    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                        sparse_sizes=2 * x.shape[:1]).t() # in case it is directed...

                # layer(x, edge_index, edge_weight)
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()



if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import DataLoader
    # Load the Cora dataset
    dataset = Planetoid(root='data', name='Cora')

    # Access the data
    data = dataset[0]

    # Print the dataset information
    print(data)

    # Create a DataLoader for the Cora dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GCN(1433,128,3,3,jk='last',node_cls=True,device='cpu')
    print (model)
    for b in dataloader:
        out = model(b.x,b.edge_index)
        break
    print (out.shape)



