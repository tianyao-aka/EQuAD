
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINEConv,GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv,GATv2Conv
import sys
# from nov.dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from torch_geometric.datasets import TUDataset

def weight_reset(m):
    if isinstance(m, nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class GIN(torch.nn.Module):
    def __init__(self,nfeat=4, nlayers=3, nhid=32,dropout=0.,pooling='attention',nclass=2, *args, **kargs):
        super(GIN, self).__init__()
        
        edge_dim=None
        dataset = kargs['dataset']
        useEdge = 0
        self.useEdge = useEdge
        self.num_features = nfeat
        
        hidden = nhid
        if 'edge_attr' in dataset[0] and useEdge:
            edge_dim = dataset[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim,hidden)
            self.conv1 = GINEConv(
                nn.Sequential(
                    Linear(dataset.num_features, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU()
                ),
                train_eps=True,edge_dim=hidden)
        else:
            self.conv1 = GINConv(
                nn.Sequential(
                    Linear(dataset.num_features, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU()
                ),
                train_eps=True)
        self.convs = torch.nn.ModuleList()
        self.dropout_val = dropout
        for i in range(nlayers - 1):
            if 'edge_attr' in dataset[0] and False:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True,edge_dim=hidden))
            else:
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True))


        self.lin1 = torch.nn.Linear(nlayers * hidden, hidden)
        self.pred_layer = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU())
        self.cls_layer = nn.Linear(hidden,nclass)
        self.pred_layer_env = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,nclass))
        if pooling=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
        elif pooling=='sum':
            self.pool = global_add_pool
        elif pooling=='mean':
            self.pool = global_mean_pool
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.apply(weight_reset)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self,x,edge_index, edge_attr = None, edge_weight=None,batch=None,output_emb = False,return_both_rep=False):
        tag = True

        x = x[:,:self.num_features]
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        
        h = F.relu(self.lin1(torch.cat(xs, dim=1)))
        x = self.pool(h,batch)
        if tag:
            if not output_emb:
                # x = self.cls_layer(self.pred_layer(x))
                x = self.pred_layer(x)
                return x 
            else:
                graph_emb = self.pred_layer(x)
                out = self.cls_layer(graph_emb)
                return out,graph_emb
        else:
            return h,x # node and graph embeddings


    def __repr__(self):
        return self.__class__.__name__


# class GIN(torch.nn.Module):
#     def __init__(self, nlayers=3, nhid=32,dropout=0.,pooling='attention',num_classes=2, *args, **kargs):
#         super(GIN, self).__init__()
        
#         edge_dim=None
#         dataset = kargs['pyg_dataset']
#         useEdge = 0
#         self.useEdge = useEdge
#         self.num_features = dataset.num_features
        
#         hidden = nhid
#         if 'edge_attr' in dataset[0] and useEdge:
#             edge_dim = dataset[0].edge_attr.shape[1]
#             self.edge_emb = nn.Linear(edge_dim,hidden)
#             self.conv1 = GINEConv(
#                 nn.Sequential(
#                     Linear(dataset.num_features, hidden),
#                     BN(hidden),
#                     ReLU(),
#                     Linear(hidden, hidden),
#                     BN(hidden),
#                     ReLU()
#                 ),
#                 train_eps=True,edge_dim=hidden)
#         else:
#             self.conv1 = GINConv(
#                 nn.Sequential(
#                     Linear(dataset.num_features, hidden),
#                     BN(hidden),
#                     ReLU(),
#                     Linear(hidden, hidden),
#                     BN(hidden),
#                     ReLU()
#                 ),
#                 train_eps=True)
#         self.convs = torch.nn.ModuleList()
#         self.dropout_val = dropout
#         for i in range(nlayers - 1):
#             if 'edge_attr' in dataset[0] and False:
#                 self.convs.append(
#                     GINEConv(
#                         Sequential(
#                             Linear(hidden, hidden),
#                             BN(hidden),
#                             ReLU(),
#                             Linear(hidden, hidden),
#                             BN(hidden),
#                             ReLU()
#                         ),
#                         train_eps=True,edge_dim=hidden))
#             else:
#                 self.convs.append(
#                     GINConv(
#                         nn.Sequential(
#                             Linear(hidden, hidden),
#                             BN(hidden),
#                             ReLU(),
#                             Linear(hidden, hidden),
#                             BN(hidden),
#                             ReLU()
#                         ),
#                         train_eps=True))
        
#         self.lin1 = torch.nn.Linear(nlayers * hidden, hidden)
#         self.pred_layer = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU())
#         self.cls_layer = nn.Linear(hidden,num_classes)
#         self.pred_layer_env = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,num_classes))
#         if pooling=='attention':
#             self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
#         elif pooling=='sum':
#             self.pool = global_add_pool
#         elif pooling=='mean':
#             self.pool = global_mean_pool
#         # self.lin2 = Linear(hidden, dataset.num_classes)
#         self.dropout = nn.Dropout(dropout)
#         self.apply(weight_reset)
#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.lin1.reset_parameters()

#     def forward(self, batch,x=None,edge_index=None,batch_idx=None,output_emb = False):
#         tag = x is None
#         if x is None:
#             x = batch.x
#             edge_index = batch.edge_index
#             batch = batch.batch
#         else:
#             batch = batch_idx

#         x = x[:,:self.num_features]
#         x = self.conv1(x, edge_index)
#         xs = [x]
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             xs += [x]
        
#         h = F.relu(self.lin1(torch.cat(xs, dim=1)))
#         x = self.pool(h,batch)
#         if tag:
#             if not output_emb:
#                 x = self.cls_layer(self.pred_layer(x))
#                 return x 
#             else:
#                 graph_emb = self.pred_layer(x)
#                 out = self.cls_layer(graph_emb)
#                 return out,graph_emb
#         else:
#             return h,x # node and graph embeddings


#     def __repr__(self):
#         return self.__class__.__name__
    

class GIN2(torch.nn.Module):
    def __init__(self, nlayers=3, nhid=32,dropout=0.,pooling='attention',num_classes=2, *args, **kargs):
        super(GIN2, self).__init__()
        
        edge_dim=None
        dataset = kargs['pyg_dataset']
        useEdge = 0
        self.useEdge = useEdge
        self.num_features = dataset.num_features
        
        hidden = nhid
        if 'edge_attr' in dataset[0] and useEdge:
            edge_dim = dataset[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim,hidden)
            self.conv1 = GINEConv(
                Sequential(
                    Linear(dataset.num_features, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU()
                ),
                train_eps=True,edge_dim=hidden)
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(dataset.num_features, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True)
        self.convs = torch.nn.ModuleList()
        self.dropout_val = dropout
        for i in range(nlayers - 1):
            if 'edge_attr' in dataset[0] and False:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True,edge_dim=hidden))
            else:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=True))
        self.lin1 = torch.nn.Linear(nlayers * hidden, hidden)
        self.pred_layer = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,num_classes))
        self.pred_layer_env = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,num_classes))
        if pooling=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
        elif pooling=='sum':
            self.pool = global_add_pool
        elif pooling=='mean':
            self.pool = global_mean_pool
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.apply(weight_reset)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, batch,x=None,edge_index=None,batch_idx=None):
        tag = x is None
        if x is None:
            x = batch.x
            edge_index = batch.edge_index
            batch = batch.batch
        else:
            batch = batch_idx

        x = x[:,:self.num_features]
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        
        h = F.relu(self.lin1(torch.cat(xs, dim=1)))
        x = self.pool(h,batch)
        if tag:
            h_label = self.pred_layer(x)
            h_env = self.pred_layer_env(x)
            return h_label,h_env
        else:
            return h,x # node and graph embeddings


    def __repr__(self):
        return self.__class__.__name__



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, Sequential, BatchNorm as BN
from torch_geometric.nn.glob import AttentionalAggregation

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class GCN(torch.nn.Module):
    def __init__(self, nlayers=3, nhid=32, dropout=0., pooling='attention', num_classes=2, *args, **kargs):
        super(GCN, self).__init__()

        dataset = kargs['pyg_dataset']
        self.num_features = dataset.num_features
        hidden = nhid

        # First GCNConv layer
        self.conv1 = GCNConv(dataset.num_features, hidden)

        # Additional GCNConv layers
        self.convs = torch.nn.ModuleList()
        self.dropout_val = dropout
        for i in range(nlayers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        # Linear layer for combining node embeddings
        self.lin1 = torch.nn.Linear(nlayers * hidden, hidden)

        # Prediction and classification layers
        self.pred_layer = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.cls_layer = nn.Linear(hidden, num_classes)
        self.pred_layer_env = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

        # Pooling method
        if pooling == 'attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Weight initialization
        self.apply(weight_reset)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, batch, x=None, edge_index=None, batch_idx=None, output_emb=False):
        tag = x is None
        if x is None:
            x = batch.x
            edge_index = batch.edge_index
            batch = batch.batch
        else:
            batch = batch_idx

        x = x[:, :self.num_features]
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)

        h = F.relu(self.lin1(torch.cat(xs, dim=1)))
        x = self.pool(h, batch)
        if tag:
            if not output_emb:
                x = self.cls_layer(self.pred_layer(x))
                return x
            else:
                graph_emb = self.pred_layer(x)
                out = self.cls_layer(graph_emb)
                return out, graph_emb
        else:
            return h, x  # node and graph embeddings

    def __repr__(self):
        return self.__class__.__name__




class GAT(torch.nn.Module):
    def __init__(self, nlayers=3, nhid=32, dropout=0., pooling='sum', num_classes=2, heads=1, *args, **kargs):
        super(GAT, self).__init__()

        dataset = kargs['pyg_dataset']
        self.num_features = dataset.num_features
        hidden = nhid

        # First GATv2Conv layer
        self.conv1 = GATv2Conv(dataset.num_features, hidden, heads=heads, dropout=dropout)

        # Additional GATv2Conv layers
        self.convs = torch.nn.ModuleList()
        for i in range(nlayers - 1):
            self.convs.append(GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout))

        # Linear layer for combining node embeddings
        self.lin1 = torch.nn.Linear(nlayers * hidden * heads, hidden)

        # Prediction and classification layers
        self.pred_layer = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.cls_layer = nn.Linear(hidden, num_classes)
        self.pred_layer_env = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

        # Pooling method
        if pooling == 'attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden * heads, 1))
        elif pooling == 'sum':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Weight initialization
        self.apply(weight_reset)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, batch, x=None, edge_index=None, batch_idx=None, output_emb=False):
        tag = x is None
        if x is None:
            x = batch.x
            edge_index = batch.edge_index
            batch = batch.batch
        else:
            batch = batch_idx

        x = x[:, :self.num_features]
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)

        h = F.relu(self.lin1(torch.cat(xs, dim=1)))
        x = self.pool(h, batch)
        if tag:
            if not output_emb:
                x = self.cls_layer(self.pred_layer(x))
                return x
            else:
                graph_emb = self.pred_layer(x)
                out = self.cls_layer(graph_emb)
                return out, graph_emb
        else:
            return h, x  # node and graph embeddings

    def __repr__(self):
        return self.__class__.__name__

class MLP(nn.Module):
    def __init__(self, input_size, nlayers=2, hidden_size=32, C=3):
        """
        Initializes the MLP.

        Args:
        input_size (int): The size of the input features.
        nlayers (int): The number of layers in the MLP.
        hidden_size (int): The size of each hidden layer.
        C (int): The number of output classes.
        """
        
        super(MLP, self).__init__()
        self.nlayers = nlayers

        layers = []
        for i in range(nlayers):
            if i == 0:
                # First layer
                layers.append(nn.Linear(input_size, hidden_size))
            elif i < nlayers - 1:
                # Hidden layers
                layers.append(nn.Linear(hidden_size, hidden_size))
            else:
                # Output layer
                layers.append(nn.Linear(hidden_size, C))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output of the MLP.
        """
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.nlayers - 1:  # Apply ReLU to all but the last layer
                x = F.relu(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    path = "data/"
    dataset = TUDataset(path, name='MUTAG')
    dataloader = DataLoader(dataset, batch_size=6)
    input_dim = max(dataset.num_features, 1)
    
    
    # gcn1 = GConv(input_dim=input_dim, nhid=64, nlayers=3).to(device)
    # gcn2 = GConv(input_dim=input_dim, nhid=64, nlayers=3).to(device)
    gcn1 = GIN(nlayers=3, nhid=32,dropout=0.,pooling='attention',pyg_dataset = dataset).to(device)
    v = next(iter(dataloader))
    val = gcn1(v.x,v.edge_index,v.batch,v)
    
    
    