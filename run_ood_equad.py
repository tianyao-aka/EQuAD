
import torch
import torch.functional as F
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
from infomax import train
from model.DualBranchContrast import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader,Data
from torch_geometric.datasets import TUDataset

from model.utils import calc_cluster_labels
from model.equad_learner import ModelTrainer
from model.data_utils import CustomDataset,DatasetWithSpuRep,get_indices_and_boolean_tensor
from model.utils import save_numpy_array_to_file,train_svm_with_calibration
from termcolor import colored


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
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from datasets.spmotif_dataset import SPMotif
from torch_geometric.transforms import BaseTransform
from drugood.datasets import build_dataset
from mmcv import Config
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='EQuAD Model Trainer Arguments')
parser.add_argument('--dataset', default='drugood_lbap_core_ec50_scaffold', type=str)
parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')

parser.add_argument('--nfeat', type=int, default=39, help='Number of features.')
parser.add_argument('--nhid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=3, help='Number of classes.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate.')
parser.add_argument('--edge_dim', type=int, default=-1, help='Edge dimension.')
parser.add_argument('--save_mem', type=bool, default=True, help='Flag to save memory.')
parser.add_argument('--jk', type=str, default='last', help='Jumping knowledge method.')
parser.add_argument('--node_cls', type=bool, default=False, help='Node classification flag.')
parser.add_argument('--pooling', type=str, default='sum', help='Pooling method.')
parser.add_argument('--with_bn', type=bool, default=False, help='Batch normalization flag.')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay for optimizer.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=128, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=50, help='Learning rate.')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
parser.add_argument('--early_stop_epochs', type=int, default=50, help='Early stopping epochs.')
parser.add_argument('--penalty', type=float, default=1e-1, help='penalty.')
parser.add_argument('--project_layer_num', type=int, default=2, help='Number of project layers.')
parser.add_argument('--base_gnn', type=str, default='gin', help='Base GNN model.')

parser.add_argument('--gamma', type=float, default=0.5, help='gamma for sample reweighting')
parser.add_argument('--valid_metric', type=str, default='acc', help='Validation metric.')
parser.add_argument('--temp', type=int, default=0.01, help='temp for model reweighting')

parser.add_argument('--device', type=int, default=0, help='Device to run the model on.')
parser.add_argument('--spu_emb_path', type=str, default='', help='load spu emb')
parser.add_argument('--ood_path', type=str, default='ood_res')
parser.add_argument('--fname_str', type=str, default='', help='additional name for folder name')
parser.add_argument('--seed', type=int, default=1) 
args = parser.parse_args()


seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = vars(args)


dataset_name = args['dataset']
if 'drugood' in dataset_name.lower():
    metric_name = 'auc'
if 'spmotif' in dataset_name.lower():
    metric_name = 'acc'
if 'ogbg' in dataset_name.lower():
    metric_name = 'auc'


if 'drugood' in dataset_name.lower():
    args['nclass'] = 2
    args["nfeat"] = 39


if 'spmotif' in dataset_name.lower():
    args['nclass'] = 3
    args["nfeat"] = 4


#! init dataset
c_labels = None
binary_c_id = None
binary_c_counts = None
workers = 2 if torch.cuda.is_available() else 0
if args['dataset'].lower().startswith('spmotif') or args['dataset'].lower().startswith('tspmotif'):
    train_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='train')
    val_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='val')
    test_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='test')
    c_labels = None
    binary_c_id = None
    binary_c_counts = None
    #! load spurious emb
    path = args['spu_emb_path']
    print (colored(f'spu_emb path:{path}','yellow'))
    if path.endswith(".pt"):
        if len(path)>4:
            spu_emb = torch.load(path)
        else:
            print (colored("Spu Emb is None. Cautious!",'yellow'))
            spu_emb = None
        
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        svm_logits,_,_ = train_svm_with_calibration(spu_emb,labels)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='spmotif',spu_rep=spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits = svm_logits)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    
    else:
        directory = os.path.dirname(args['spu_emb_path'])
        
        # Find all training and validation file paths
        train_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_train" in f]
        valid_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_valid" in f]
        
        # Create a dictionary to match train and valid files by their common substring (e.g., epoch)
        train_files = {os.path.basename(f).replace("graph_emb_train_", "").replace(".pt", ""): f for f in train_paths}
        valid_files = {os.path.basename(f).replace("graph_emb_valid_", "").replace(".pt", ""): f for f in valid_paths}
        
        # Ensure we have corresponding pairs
        common_keys = train_files.keys() & valid_files.keys()
        
        emb_pairs = [(torch.load(train_files[k]), torch.load(valid_files[k])) for k in common_keys]
        
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        valid_labels = torch.cat([val_dataset[i].y.view(-1,) for i in range(len(val_dataset))])
        
        svm_logits_list = []
        train_score_list = []
        valid_score_list = []
        for spu_emb, spu_emb_valid in tqdm(emb_pairs):
            svm_logits, train_score, valid_score = train_svm_with_calibration(spu_emb, labels, spu_emb_valid, valid_labels, "accuracy")
            svm_logits_list.append(svm_logits)
            train_score_list.append(train_score)
            valid_score_list.append(valid_score)
            print (f'training acc:{train_score},valid acc:{valid_score}')

        # Sort by valid_score_list and get the indices of the lowest 5 scores
        sorted_indices = sorted(range(len(valid_score_list)), key=lambda i: valid_score_list[i])[:5]
        
        # Get the corresponding spu_emb for the lowest 5 scores
        lowest_5_spu_emb = torch.stack([emb_pairs[i][0] for i in sorted_indices])
        lowest_5_svm_logits = torch.stack([svm_logits_list[i] for i in sorted_indices])
        lowest_5_spu_emb = torch.permute(lowest_5_spu_emb,(1,0,2))
        lowest_5_svm_logits = torch.permute(lowest_5_svm_logits,(1,0,2))
        # Get the valid scores for the lowest 5 scores and scale them
        lowest_5_valid_scores = torch.tensor([valid_score_list[i] for i in sorted_indices]) * -1.0
        scaled_scores = torch.softmax(lowest_5_valid_scores / args['temp'], dim=0)
        args['model_weights'] = scaled_scores  #! (5,)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]    
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='spmotif',spu_rep=lowest_5_spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits =lowest_5_svm_logits,num_spu_emb=5)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
        
    
    valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    args['nclass'] = 3
    metric_name='acc'
    args['valid_metric'] = metric_name


elif args['dataset'].lower().startswith('drugood'):
    #drugood_lbap_core_ic50_assay.json
    metric_name='auc'
    args['valid_metric'] = metric_name
    config_path = os.path.join("configs", args["dataset"] + ".py")
    cfg = Config.fromfile(config_path)
    root = os.path.join(args["root"],"DrugOOD")
    train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args["dataset"], mode="train")
    val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args["dataset"], mode="ood_val")
    test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args["dataset"], mode="ood_test")
    #! load spurious emb
    path = args['spu_emb_path']
    print (colored(f'spu_emb path:{path}','yellow'))
    if path.endswith(".pt"):
        if len(path)>4:
            spu_emb = torch.load(path)
        else:
            print (colored("Spu Emb is None. Cautious!",'yellow'))
            spu_emb = None
            
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        svm_logits,_,_ = train_svm_with_calibration(spu_emb,labels)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='drugood',spu_rep=spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits = svm_logits)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='drugood',spu_rep=spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits = svm_logits)

        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)

    else:
        directory = os.path.dirname(args['spu_emb_path'])
        
        # Find all training and validation file paths
        train_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_train" in f]
        valid_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_valid" in f]
        
        # Create a dictionary to match train and valid files by their common substring (e.g., epoch)
        train_files = {os.path.basename(f).replace("graph_emb_train_", "").replace(".pt", ""): f for f in train_paths}
        valid_files = {os.path.basename(f).replace("graph_emb_valid_", "").replace(".pt", ""): f for f in valid_paths}
        
        # Ensure we have corresponding pairs
        common_keys = train_files.keys() & valid_files.keys()
        
        emb_pairs = [(torch.load(train_files[k]), torch.load(valid_files[k])) for k in common_keys]
        
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        valid_labels = torch.cat([val_dataset[i].y.view(-1,) for i in range(len(val_dataset))])
        
        svm_logits_list = []
        train_score_list = []
        valid_score_list = []
        for spu_emb, spu_emb_valid in emb_pairs:
            svm_logits, train_score, valid_score = train_svm_with_calibration(spu_emb, labels, spu_emb_valid, valid_labels, "auc")
            svm_logits_list.append(svm_logits)
            train_score_list.append(train_score)
            valid_score_list.append(valid_score)

        # Sort by valid_score_list and get the indices of the lowest 5 scores
        sorted_indices = sorted(range(len(valid_score_list)), key=lambda i: valid_score_list[i])[:5]
        
        # Get the corresponding spu_emb for the lowest 5 scores
        lowest_5_spu_emb = torch.stack([emb_pairs[i][0] for i in sorted_indices])
        lowest_5_svm_logits = torch.stack([svm_logits_list[i] for i in sorted_indices])
        lowest_5_spu_emb = torch.permute(lowest_5_spu_emb,(1,0,2))
        lowest_5_svm_logits = torch.permute(lowest_5_svm_logits,(1,0,2))
        # Get the valid scores for the lowest 5 scores and scale them
        lowest_5_valid_scores = torch.tensor([valid_score_list[i] for i in sorted_indices]) * -1.0
        scaled_scores = torch.softmax(lowest_5_valid_scores / args['temp'], dim=0)
        args['model_weights'] = scaled_scores  #! (5,)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]    
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='spmotif',spu_rep=lowest_5_spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits =lowest_5_svm_logits,num_spu_emb=5)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    print ('len of test dataset of:',args['dataset'])
    print (len(test_dataset))


elif args["dataset"].lower().startswith('ogbg'):
    def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

    if 'ppa' in args["dataset"].lower():
        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"], transform=add_zeros)
        input_dim = -1
        num_classes = dataset.num_classes
    else:
        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"])
        input_dim = 1
        num_classes = dataset["num_tasks"]
        args['nclass'] = dataset["num_tasks"]
        args["nfeat"] = 1
    split_idx = dataset.get_idx_split()
    metric_name = 'auc' #! watch out this!
    args['valid_metric'] = metric_name

    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    path = args['spu_emb_path']
    if path.endswith(".pt"):
        if len(path)>4:
            spu_emb = torch.load(path)
        else:
            print (colored("Spu Emb is None. Cautious!",'yellow'))
            spu_emb = None
            
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        svm_logits,_,_ = train_svm_with_calibration(spu_emb,labels)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='ogbg',spu_rep=spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits = svm_logits)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='ogbg',spu_rep=spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits = svm_logits)

        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)

    else:
        directory = os.path.dirname(args['spu_emb_path'])
        
        # Find all training and validation file paths
        train_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_train" in f]
        valid_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt") and "graph_emb_valid" in f]
        
        # Create a dictionary to match train and valid files by their common substring (e.g., epoch)
        train_files = {os.path.basename(f).replace("graph_emb_train_", "").replace(".pt", ""): f for f in train_paths}
        valid_files = {os.path.basename(f).replace("graph_emb_valid_", "").replace(".pt", ""): f for f in valid_paths}
        
        # Ensure we have corresponding pairs
        common_keys = train_files.keys() & valid_files.keys()
        
        emb_pairs = [(torch.load(train_files[k]), torch.load(valid_files[k])) for k in common_keys]
        
        labels = torch.cat([train_dataset[i].y.view(-1,) for i in range(len(train_dataset))])
        valid_labels = torch.cat([val_dataset[i].y.view(-1,) for i in range(len(val_dataset))])
        
        svm_logits_list = []
        train_score_list = []
        valid_score_list = []
        for spu_emb, spu_emb_valid in emb_pairs:
            svm_logits, train_score, valid_score = train_svm_with_calibration(spu_emb, labels, spu_emb_valid, valid_labels, "auc")
            svm_logits_list.append(svm_logits)
            train_score_list.append(train_score)
            valid_score_list.append(valid_score)

        # Sort by valid_score_list and get the indices of the lowest 5 scores
        sorted_indices = sorted(range(len(valid_score_list)), key=lambda i: valid_score_list[i])[:5]
        
        # Get the corresponding spu_emb for the lowest 5 scores
        lowest_5_spu_emb = torch.stack([emb_pairs[i][0] for i in sorted_indices])
        lowest_5_svm_logits = torch.stack([svm_logits_list[i] for i in sorted_indices])
        lowest_5_spu_emb = torch.permute(lowest_5_spu_emb,(1,0,2))
        lowest_5_svm_logits = torch.permute(lowest_5_svm_logits,(1,0,2))
        # Get the valid scores for the lowest 5 scores and scale them
        lowest_5_valid_scores = torch.tensor([valid_score_list[i] for i in sorted_indices]) * -1.0
        scaled_scores = torch.softmax(lowest_5_valid_scores / args['temp'], dim=0)
        args['model_weights'] = scaled_scores  #! (5,)
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]    
        train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='spmotif',spu_rep=lowest_5_spu_emb,cluster_id=c_labels,binary_cluster_id=binary_c_id,binary_cluster_count=binary_c_counts,svm_logits =lowest_5_svm_logits,num_spu_emb=5)
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)


    valid_loader = DataLoader(dataset[split_idx["valid"]],
                                batch_size=args["batch_size"],
                                shuffle=False,
                                num_workers=workers)
    test_loader = DataLoader(dataset[split_idx["test"]],
                                batch_size=args["batch_size"],
                                shuffle=False,
                                num_workers=workers)

args['device'] = 'cpu' if not torch.cuda.is_available() else args['device']
model = ModelTrainer(**args)
print ('fit model')
model.fit(train_loader,valid_loader,test_loader,epochs=args["epochs"])
model.load_state_dict(model.best_states)
res = model.valid_metric_list
res = sorted(res,key = lambda x:x[0],reverse=True)
val_score,test_score = res[0]
res = np.array([val_score,test_score])

print ('best res:',res)



