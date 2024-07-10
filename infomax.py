
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
# from GCL.models import DualBranchContrast
from model.DualBranchContrast import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader,Data
from torch_geometric.datasets import TUDataset
from model.gin import GIN
from model.data_utils import CustomDataset,get_indices_and_boolean_tensor
from termcolor import colored
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
# from utils.trainingUtils import * 
# from utils.functionalUtils import *

from copy import deepcopy
import argparse
import os
import sys
import warnings
import random
from datasets import spmotif_dataset
from torch_geometric.data import Batch

from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from torch_geometric.transforms import BaseTransform
from drugood.datasets import build_dataset
from mmcv import Config

warnings.filterwarnings("ignore")


class FeatureSelector(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        """
        Concatenate random features to each node in the graph. If node features
        do not exist, new random features are assigned as node features.

        Parameters:
        - data (torch_geometric.data.Data): The graph data object.

        Returns:
        - torch_geometric.data.Data: The modified graph data object with added features.
        """
        x = data.x[:,:4]
        data.x = x
        return data

def save_tensor_to_file(X, fpath, name):
    """
    Save a PyTorch tensor X to a specified file path and name.
    
    Parameters:
    - X (torch.Tensor): Tensor of shape (N, D) to be saved.
    - fpath (str): The directory where the tensor should be saved.
    - name (str): The name of the file to save the tensor as.
    
    Returns:
    None
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    # Full path to the file
    full_path = os.path.join(fpath, name)
    
    # Remove the file if it already exists
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Save the tensor to the file
    torch.save(X, full_path)



def adjust_node_indices(batch_node_indices, node_indices_lengths, batch_ptr):
    """
    Adjust the node indices with offsets using lengths of node indices and batch pointers.

    Parameters:
    batch_node_indices (torch.Tensor): Tensor containing node indices of the batch.
    node_indices_lengths (torch.Tensor): Tensor containing the lengths of node indices for each graph in the batch.
    batch_ptr (torch.Tensor): Tensor containing pointers to the start of each graph in the batch.

    Returns:
    torch.Tensor: Adjusted node indices with appropriate offsets.
    """
    adjusted_node_indices = []
    start_idx = 0

    for i, length in enumerate(node_indices_lengths):
        end_idx = start_idx + length
        indices = batch_node_indices[start_idx:end_idx]
        offset = batch_ptr[i]
        adjusted_indices = indices + offset
        adjusted_node_indices.append(adjusted_indices)
        start_idx = end_idx

    return torch.cat(adjusted_node_indices)


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)



class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
        z1, g1 = self.gcn1(x1, edge_index1, batch=batch,edge_attr=None,return_both_rep=True)
        z2, g2 = self.gcn2(x2, edge_index2, batch=batch,edge_attr=None,return_both_rep=True)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device) if torch.cuda.is_available() else data.to('cpu')
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
    
        h1, h2, g1, g2 = encoder_model(data)
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=data.batch,data=data) #! new argument data for adjust pos masks
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader,numPoints = 10,num_trials=15):
    results = []
    encoder_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device) if torch.cuda.is_available() else data.to('cpu')
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, _, g1, g2 = encoder_model(data)
            x.append(g1 + g2)
            y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    kmeans_acc_result = []
    #! use linear-SVM as augmentation
    # msg = train_and_evaluate_svm(x_np,y_np,K=numPoints,S=int(0.6*numPoints),num_trials=num_trials)
    return None,x.cpu()


if __name__ == '__main__':
    seed=1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # provide a parser for the command line
    parser = argparse.ArgumentParser()
    # add augument for string arguments

    parser.add_argument('--hidden_dims',type=int,default=64)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--edge_dim',type=int,default=-1) # -1 means not using edge attr
    parser.add_argument('--dataset',type=str,default="Graph-SST5")
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    parser.add_argument('--SSL',type=str,default="MVGRL")
    parser.add_argument('--device_id',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=1)
    # parser.add_argument('--fold_index',type=int,default=0)
    
    args = parser.parse_args()
    args = vars(args)
    if "ogbg" in args['dataset'].lower():
        name = args['dataset'].lower().replace("-","_")
    
    
    embeddingWritingPath = f"experiment_results/SSL_embedding/{args['dataset']}/infomax/hidden_dims_{args['hidden_dims']}_num_layers_{args['num_layers']}/"
    if os.path.exists(embeddingWritingPath):
        sys.exit(f'------------------------already finished running-------------------------')

    
    hidden_dim = args["hidden_dims"]
    num_layers = args["num_layers"]
    dataset_name = args["dataset"]
    device = torch.device(f'cuda:{args["device_id"]}') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 1 if torch.cuda.is_available() else 0
    print (colored(f"using device {device}",'red','on_white'))
    path = "data/"
    
    
    if args['dataset'].lower().startswith('drugood'):
        #drugood_lbap_core_ic50_assay.json
        config_path = os.path.join("configs", args["dataset"] + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args["root"],"DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args["dataset"], mode="train")
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        
        
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args["dataset"], mode="ood_val")
        valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        input_dim = 39
        edge_dim = 10
        num_classes = 2
        input_dim = max(train_dataset.num_features, 1)
    
    
    elif args["dataset"].lower().startswith('ogbg'):

        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"])
        print (dataset)
        input_dim = 1
        args['nclass'] = 2
        split_idx = dataset.get_idx_split()
        metric_name = 'auc' #! watch out this!
        args['valid_metric'] = metric_name
        ### automatic evaluator. takes dataset name as input
        train_dataset = dataset[split_idx["train"]]
        valid_dataset = dataset[split_idx["valid"]]
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        #! process data
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        valid_loader = DataLoader(valid_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        input_dim = train_dataset[0].x.shape[1]
        edge_dim = train_dataset[0].edge_attr.shape[1]
        num_classes = 2
        
        
    elif 'spmotif' in args['dataset'].lower():
        train_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='train')
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        val_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='val')
        
        #! process dataset to custom dataset
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]

        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        # test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        input_dim=4
        num_classes = 3

        #! debug
        # dat = next(iter(train_loader))
        # print (dat)
        # print (dat.node_label)
        # print (dat.node_indices)
        # print (dat.node_indices_len)
        # print (dat.node_mask)
        # print (dat.ptr)
        # adj_node_indices = adjust_node_indices(dat.node_indices, dat.node_indices_len, dat.ptr)
        # dat.adj_node_indices = adj_node_indices
        # print (dat)
        # sys.exit()

    aug1 = A.Identity()
    aug2 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2, use_cache=False)
    gcn1 = GIN(nfeat=input_dim, nhid=hidden_dim, nclass=num_classes, nlayers=num_layers,edge_dim= args['edge_dim'], dropout=0.1, jk='last',node_cls=False,pooling='sum',dataset=args["dataset"].lower()).to(device)
    gcn2 = GIN(nfeat=input_dim, nhid=hidden_dim, nclass=num_classes, nlayers=num_layers,edge_dim= args['edge_dim'], dropout=0.1, jk='last',node_cls=False,pooling='sum',dataset=args["dataset"].lower()).to(device)
    mlp1 = FC(input_dim=hidden_dim, output_dim=hidden_dim)
    mlp2 = FC(input_dim=hidden_dim, output_dim=hidden_dim)
    encoder_model = Encoder(gcn1=gcn1, gcn2=gcn2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L',device=device).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=2e-3)

    
    #! use loss as the metric to select model
    min_loss = 10000.0
    tr_emb_dict = {}
    val_emb_dict = {}
    with tqdm(total=args['epochs'], desc='(T)') as pbar:
        for epoch in range(1, args['epochs']+1):
            loss = train(encoder_model, contrast_model, train_loader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch%3==0 and epoch>0:
                _,tr_emb = test(encoder_model, full_train_loader,num_trials=15,numPoints=10)
                _,val_emb = test(encoder_model, valid_loader,num_trials=15,numPoints=10)
                tr_emb_dict[f'epoch_{epoch}']=tr_emb
                val_emb_dict[f'epoch_{epoch}']=val_emb
            
            if loss < min_loss:
                min_loss = loss
                best_epoch = epoch
                best_model = deepcopy(encoder_model)
        _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
        tr_emb_dict[f'epoch_{best_epoch}']=tr_emb
        val_emb_dict[f'epoch_{best_epoch}']=val_emb
    
    # ! use 10-fold to evaluate the performance of linear SVM
    
    # _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
    # _,val_emb = test(best_model, valid_loader,num_trials=15,numPoints=10)
    # _,test_emb = test(best_model, test_loader,num_trials=15,numPoints=10)
    # write_results_to_file(resultWritingPath,'result.txt',msg)
    for k in tr_emb_dict:
        save_tensor_to_file(tr_emb_dict[k].cpu(),embeddingWritingPath,f'graph_emb_train_{k}.pt')
        save_tensor_to_file(val_emb_dict[k].cpu(),embeddingWritingPath,f'graph_emb_valid_{k}.pt')
    # print ('success')
    
    
    
    