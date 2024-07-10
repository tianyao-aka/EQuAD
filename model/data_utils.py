import os
import numpy as np
import pandas as pd
import re
import torch
from torch_geometric.data import Dataset, Data

def parse_dir_name(dir_name):
    """
    Parse the directory name to extract parameters and their values.
    
    Args:
    - dir_name (str): The directory name to parse.
    
    Returns:
    - dict: A dictionary where keys are parameter names and values are parameter values.
    """
    # Split the directory name by '_' and parse
    parts = dir_name.split('_')
    params = {}
    for i in range(0, len(parts), 2):
        key = parts[i]
        try:
            value = parts[i + 1]
            # Attempt to convert numeric values
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            value = value  # Keep as string if conversion fails
        except IndexError:
            continue  # Skip if there's no value for a key
        params[key] = value
    return params

def load_dataframes(root_dir):
    """
    Load numpy arrays from subdirectories and return a pandas DataFrame.
    
    Args:
    - root_dir (str): The root directory to search recursively.
    
    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a subdirectory's attributes and loaded numpy array.
    """
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                # Load the numpy array
                npy_path = os.path.join(subdir, file)
                npy_array = np.load(npy_path)
                
                # Parse the directory name to get parameters
                dir_name = os.path.relpath(subdir, root_dir)
                params = parse_dir_name(dir_name.replace('/', '_'))
                
                # Add the numpy array to params dict
                params['numpyArray'] = npy_array
                
                # Append to data list
                data.append(params)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def get_indices_and_boolean_tensor(tensor):
    """
    Given a 1D tensor, find the indices of elements equal to 1 and return
    a boolean tensor of the same size as the indices tensor, with all elements set to True.

    Parameters:
    tensor (torch.Tensor): The input 1D tensor.

    Returns:
    torch.Tensor: A boolean tensor of the same size as the indices tensor with all True values.
    """
    # Find the indices where the tensor elements are equal to 1
    indices = torch.nonzero(tensor == 1).flatten()

    # Create a boolean tensor of the same size as the indices tensor, with all elements set to True
    boolean_tensor = torch.ones(indices.size(0), dtype=torch.bool)
    return indices, boolean_tensor


class CustomDataset(Dataset):
    def __init__(self, data_list, dataset_name='spmotif', node_indices=None,node_mask=None, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(transform=transform, pre_transform=pre_transform)
        self.dset_name = dataset_name
        self.data_list = data_list
        self.node_indices = node_indices
        self.node_mask = node_mask

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]

        # Add the new attribute to the Data object
        if 'spmotif' in self.dset_name.lower():
            indices, boolean_tensor = get_indices_and_boolean_tensor(data.node_label)
            data.node_indices = indices
            data.node_indices_len = torch.tensor([len(indices)])
            data.node_mask = boolean_tensor
        else:
            data.node_indices = self.node_indices[idx]
            data.node_indices_len = len(data.node_indices)
            data.node_mask = self.node_mask[idx]
        return data



class DatasetWithSpuRep(Dataset):
    def __init__(self, data_list, dataset_name='spmotif', spu_rep=None,cluster_id=None,binary_cluster_id=None,binary_cluster_count = None,svm_logits = None,intra_cluster_pred_logits=None, intra_cluster_labels=None,sample_weights=None, transform=None, pre_transform=None,num_spu_emb=1):
        super(DatasetWithSpuRep, self).__init__(transform=transform, pre_transform=pre_transform)
        self.dset_name = dataset_name
        self.data_list = data_list
        self.spu_rep = spu_rep
        self.cluster_id = cluster_id
        self.binary_cluster_id = binary_cluster_id
        self.binary_cluster_count = binary_cluster_count
        self.svm_logits = svm_logits
        self.intra_cluster_pred_logits = intra_cluster_pred_logits
        self.intra_cluster_labels = intra_cluster_labels
        self.sample_weights = sample_weights
        self.num_emb = num_spu_emb

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        if self.spu_rep is not None:
            spu_rep = self.spu_rep[idx].view(1,self.num_emb,-1) if self.num_emb>1 else self.spu_rep[idx].view(1,-1)
            data.spu_rep = spu_rep
        if self.cluster_id is not None:
            data.cluster_id = torch.tensor(self.cluster_id[idx]).view(-1,)
        if self.binary_cluster_id is not None:
            data.binary_cluster_id = torch.tensor(self.binary_cluster_id[idx]).view(-1,)
        if self.binary_cluster_count is not None:
            data.binary_cluster_count = torch.tensor(self.binary_cluster_count[idx]).view(1,-1)
        if self.svm_logits is not None:
            data.svm_logits = torch.tensor(self.svm_logits[idx]).view(1,self.num_emb,-1) if self.num_emb>1 else torch.tensor(self.svm_logits[idx]).view(1,-1)
        if self.intra_cluster_pred_logits is not None:
            data.intra_cluster_pred_logits = torch.tensor(self.intra_cluster_pred_logits[idx]).view(1,-1)
        if self.intra_cluster_labels is not None:
            data.intra_cluster_labels = torch.tensor(self.intra_cluster_labels[idx]).view(-1,)
        if self.sample_weights is not None:
            data.sample_weights = torch.tensor(self.sample_weights[idx]).view(-1,)
        return data
    
    
    
    