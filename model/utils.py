import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch_geometric.utils import k_hop_subgraph
from sklearn.cluster import KMeans

import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.transforms import BaseTransform
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
import os




def get_model_gradients_vector(model, loss_val,retain_graph=False):
    """
    Compute gradients of the given loss value with respect to all parameters in the PyTorch model,
    and concatenate these gradients into a single vector.

    Parameters:
    - model: The PyTorch model with respect to whose parameters the gradients will be computed.
    - loss_val: The scalar loss value for which gradients are to be computed.

    Returns:
    A single tensor vector containing all the gradients concatenated.
    """
    
    # Ensure the model's parameters are ready for gradient computation
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Compute gradients of loss_val with respect to model parameters
    grads = torch.autograd.grad(loss_val, params,retain_graph=retain_graph)
    
    # Flatten and concatenate all gradients into a single vector
    gradients_vector = torch.cat([grad.view(-1) for grad in grads])
    
    return gradients_vector



class DataAugLoss(nn.Module):
    def __init__(self, threshold=0.5, high_penalty=4.0, low_penalty=1.0):
        super(DataAugLoss, self).__init__()
        self.threshold = threshold
        self.high_penalty = high_penalty
        self.low_penalty = low_penalty

    def forward(self, input):
        # a = torch.sum(inputs==0)
        # b = torch.sum(inputs==1)
        # c = len(inputs)
        # print (a,b,c)
        # Ensure inputs are in the right shape and compute the condition

        # Compute losses for both conditions
        if input >= self.threshold:
            loss = self.high_penalty * (input - self.threshold)
        else:
            loss = self.low_penalty * (self.threshold - input)
        return loss
    

import torch
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

# class modInfoNCE(Loss):
#     def __init__(self, tau):
#         super(modInfoNCE, self).__init__()
#         self.tau = tau

#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
#         sim = _similarity(anchor, sample) / self.tau
#         exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
#         log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = log_prob * pos_mask
#         loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
#         return -loss
    


class modInfoNCE(nn.Module):
    def __init__(self, tau):
        super(modInfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, neg_mask):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss  # Returning mean loss to be consistent with PyTorch loss functions


class WrapperModel(nn.Module):
    def __init__(self, *models):
        super(WrapperModel, self).__init__()
        self.dataAug_gnn = models[0]
        self.edge_linear = models[1]
        self.gnn = models[2]
        self.encoder_model = models[3]
        self.contrast_model_non_agg = models[4]
        # self.meta_loss_mlp = models[5]
        self.ssl_header = models[5]
        self.cls_header = models[6]
        self.featsMask = models[7]
        self.meta_loss_mlp = models[8]
    def forward(self):
        pass
    
    
def check_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter '{name}' has gradients.")
        else:
            print(f"Parameter '{name}' does not have gradients.")
            
            
def compare_model_params(model1: nn.Module, model2: nn.Module, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Compare parameters of two models to check if they are close enough.

    Parameters:
    - model1 (nn.Module): The first model to compare.
    - model2 (nn.Module): The second model to compare.
    - rtol (float): The relative tolerance parameter (default: 1e-05).
    - atol (float): The absolute tolerance parameter (default: 1e-08).

    Returns:
    - bool: True if all parameters of the two models are close enough, False otherwise.
    """
    # Extract parameters from both models and flatten them into tensors
    params1 = torch.cat([p.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.view(-1) for p in model2.parameters()])

    # Check if the flattened parameter tensors are close enough
    return torch.allclose(params1, params2, rtol=rtol, atol=atol)



def show_model_gradients(model):
    """
    Prints the gradients of all parameters in a PyTorch model for every module.
    
    Parameters:
    - model (nn.Module): The model whose gradients are to be displayed.
    """
    for module_name, module in model.named_modules():
        print(f"Module: {module_name} ({module.__class__.__name__})")
        for param_name, param in module.named_parameters(recurse=True):
            if param.grad is not None:
                print(f"  Param: {param_name}, Grad: {torch.sum(param.grad**2)}")
            else:
                print(f"  Param: {param_name}, Grad: None")
                
    
def save_numpy_array_to_file(array, file_path, file_name):
    """
    Save a NumPy array to a specified file path and file name.
    
    Parameters:
    - array (np.ndarray): The NumPy array to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the array to a .npy file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.npy")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the NumPy array to the file
    np.save(full_file_path, array)


def save_tensors_to_file(tensors, file_path, file_name):
    """
    Save a list of PyTorch tensors to a specified file path and file name.
    
    Parameters:
    - tensors (list): A list of PyTorch tensors to be saved.
    - file_path (str): The directory where the file will be saved.
    - file_name (str): The name of the file (without extension).
    
    Output:
    - None: The function saves the tensors to a .pt file at the specified location.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Full path to the file
    full_file_path = os.path.join(file_path, f"{file_name}.pt")
    
    # Delete the file if it already exists
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
    
    # Save the list of tensors to the file
    torch.save(tensors, full_file_path)



def extract_k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes, x, y):
    """
    Extracts the k-hop subgraph of a node, including node features and labels.
    
    Parameters:
    - node_idx (int): The index of the central node.
    - num_hops (int): The number of hops to consider for the neighborhood.
    - edge_index (Tensor): The edge index tensor of the whole graph.
    - num_nodes (int): The total number of nodes in the whole graph.
    - x (Tensor): The node feature matrix of the whole graph.
    - y (Tensor): The node labels of the whole graph.
    
    Returns:
    - sub_data (Data): A PyG Data object representing the extracted subgraph, including node features and labels.
    """
    # Extract the k-hop subgraph around the specified node
    sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    
    # Extract features and labels for nodes in the subgraph
    sub_x = x[sub_nodes]
    sub_y = y[node_idx].view(-1,)
    
    # Create a subgraph data object including features and labels
    sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
    return sub_data



def total_variation_distance(v):
    """
    Calculate the Total Variation distance between a given probability distribution tensor v
    and the uniform distribution.

    Parameters:
    - v (torch.Tensor): A 1D tensor of shape (N,) representing a probability distribution.

    Returns:
    - float: The Total Variation distance.
    """
    # Number of elements in v
    N = v.shape[0]
    
    # Uniform distribution tensor with the same shape as v
    uniform = torch.full_like(v, 1/N)
    
    # Calculate the Total Variation distance
    tv_distance = 0.5 * torch.sum(torch.abs(v - uniform))
    
    return tv_distance



class FeatureSelect(BaseTransform):
    def __init__(self, nfeats):
        """
        Initialize the transformation with the number of features to retain.

        Parameters:
        - feats (int): The number of features to retain from the beginning of the feature matrix.
        """
        self.nfeats = nfeats

    def __call__(self, data):
        """
        Retain only the first 'feats' features of the node feature matrix 'data.x'.
        Parameters:
        - data (torch_geometric.data.Data): The graph data object.
        Returns:
        - torch_geometric.data.Data: The modified graph data object with the node feature matrix sliced.
        """
        
        # Check if 'data.x' exists and has enough features
        data.x = data.x[:, :self.nfeats]
        return data



def calc_cluster_labels(emb,y,num_classes=3,num_clusters = 3):
    # init a zero np arrays with the shape of y
    cluster_labels = np.zeros_like(y)
    N = emb.shape[0]
    for c in range(num_classes):
        emb_c = emb[y==c]
        if N<4000:
            kmeans = KMeans(n_clusters=num_clusters).fit(emb_c)
        else:
            print ('use more efficient Kmeans algorithm')
            kmeans = KMeans(n_clusters=num_clusters,n_init=5,tol=1e-3,max_iter=200).fit(emb_c)
        cids = kmeans.labels_
        cluster_labels[y==c] = cids
    return torch.tensor(cluster_labels)
    
    

def train_svm_with_calibration(X, y, X_test=None, y_test=None, eval_method='accuracy'):
    """
    Trains an SVM model with probability calibration and returns logits along with evaluation scores.

    Parameters:
    X (torch.Tensor or np.ndarray): Input features of shape (N, D).
    y (torch.Tensor or np.ndarray): Target labels of shape (N,).
    X_test (torch.Tensor or np.ndarray, optional): Test input features of shape (N_test, D). Default is None.
    y_test (torch.Tensor or np.ndarray, optional): Test target labels of shape (N_test,). Default is None.
    eval_method (str): Evaluation method ('accuracy' or 'auc').

    Returns:
    torch.Tensor: Logits of shape (N, C) where C is the number of classes.
    float: Evaluation score on training data.
    float or None: Evaluation score on test data if X_test and y_test are provided, otherwise None.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if X_test is not None and isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()
    if y_test is not None and isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    # Ensure the target labels are of integer type
    y = y.astype(int)
    if y_test is not None:
        y_test = y_test.astype(int)

    # Train SVM with probability calibration
    linear_svm = LinearSVC()
    calibrated_svm = CalibratedClassifierCV(linear_svm, method='sigmoid')
    calibrated_svm.fit(X, y)

    # Get probability estimates (logits)
    logits = calibrated_svm.predict_proba(X)

    # Evaluate the performance on training data
    if eval_method == 'accuracy':
        train_score = accuracy_score(y, calibrated_svm.predict(X))
    elif eval_method == 'auc':
        train_score = roc_auc_score(y, calibrated_svm.predict_proba(X), multi_class='ovr')

    # Evaluate the performance on test data if provided
    if X_test is not None and y_test is not None:
        if eval_method == 'accuracy':
            test_score = accuracy_score(y_test, calibrated_svm.predict(X_test))
        elif eval_method == 'auc':
            test_score = roc_auc_score(y_test, calibrated_svm.predict_proba(X_test), multi_class='ovr')
    else:
        test_score = 0.0

    return torch.tensor(logits), train_score,test_score


