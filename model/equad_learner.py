from sklearn import svm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model.gin import GIN
# from torch_geometric.nn import GIN, GCN
from torchmetrics import Accuracy, AUROC

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
import random
import string
from termcolor import colored
from copy import deepcopy
import sys



class ModelTrainer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, edge_dim=-1, save_mem=True, jk='last', 
                 node_cls=False, pooling='sum', with_bn=False, weight_decay=5e-6, lr=1e-3, adapt_lr=1e-4, 
                 patience=50, early_stop_epochs=5, penalty=0.1, project_layer_num=2,
                 base_gnn='gin', kernel_method='rbf', sigma=0.1, valid_metric='acc', device='cpu', **args):
        super(ModelTrainer, self).__init__()
        
        self.delta_acc_arr = None
        self.valid_test_delta_acc_list = []
        self.gamma = args.get("gamma",0.5)
        dataset_name = args.get('dataset')
        self.model_weights = args.get('model_weights',None)
        
        if base_gnn == 'gin':
            self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout, edge_dim=edge_dim, jk=jk, node_cls=node_cls, 
                           pooling=pooling, with_bn=with_bn, weight_decay=weight_decay,dataset = dataset_name)
            self.gnn.to(device)
        elif base_gnn == 'gcn':
            self.gnn = GCN(nfeat, nhid, nclass, nlayers=nlayers, dropout=dropout, save_mem=save_mem, jk=jk, 
                           node_cls=node_cls, pooling=pooling, with_bn=with_bn, weight_decay=weight_decay)
            self.gnn.to(device)
        self.cls_header = self.create_mlp(nhid, nhid, nclass, project_layer_num).to(device)
        self.svm_logits_cls_header = self.create_mlp(nhid, nhid, nclass, project_layer_num).to(device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.penalty = penalty
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.metric_func = (Accuracy(task='multiclass', num_classes=nclass, top_k=1).to(device) 
                            if valid_metric == 'acc' else AUROC(task='binary').to(device))
        self.metric_name = valid_metric
        
        
        self.balance_sampling = args.get("balance_sampler",False)
        self.intra_class_clustering = args.get("intra_clsuter_labels",False)
        self.intra_class_penalty = args.get("intra_cluster_penalty",0.)
        if self.intra_class_clustering:
            pass
        
        print (colored(f'use balance sampling or not: {self.balance_sampling}','red'))
        print (colored(f'use intra class clustering or not: {self.intra_class_clustering}, penalty is:{self.intra_class_penalty}','red'))
        
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.valid_metric_list = []
        self.meta_valid_metric_list = []
        self.best_valid_metric = -1.
        self.test_metric = -1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement = 0
        self.stop_training = False

    
    def create_mlp(self, input_dim, hidden_dim, output_dim, num_layers, cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def parameters(self):
        return list(self.gnn.parameters()) + list(self.cls_header.parameters())
    

    def train_single_step(self,data):
        # self.encoder_model.train()
        data = data.to(self.device)
        y = data.y
        svm_logits_targets = data.svm_logits
        size = len(y)
        g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,return_both_rep=False)
        logits = self.cls_header(g)
        svm_pred_logits = self.svm_logits_cls_header(g)
        loss = self.ce_loss(logits, y.long())
        
        if self.penalty>0:
            spu_loss = []
            if self.model_weights is None:
                svm_pred_loss = self.cross_entropy_with_soft_targets(svm_pred_logits,svm_logits_targets,y.long())
            else:
                # have 5 model weights and svm_targets for prediction
                for i in range(5):
                    spu_loss.append(self.cross_entropy_with_soft_targets(svm_pred_logits,svm_logits_targets[:,i,:],y.long()))
                spu_loss = torch.tensor(spu_loss)
                svm_pred_loss = torch.sum(self.model_weights*spu_loss)
    
        return loss,svm_pred_loss

    def fit(self,train_loader,valid_dloader,test_dloader=None,epochs=50):
        for e in range(epochs):
            if e<=10:
                self.epochs_since_improvement=0
                self.stop_training=False
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            # if e==10 and self.pe>0:
            #     self.useAutoAug,self.encoder_model.learnable_aug,self.epochs_since_improvement,self.edge_uniform_penalty,self.penalty,self.edge_penalty = state
            #     self.best_valid_metric = 0.
            if self.stop_training:
                break
            total_losses = 0.
            total_svmPred_loss = 0.
            steps = 0
            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                erm_loss, svm_pred_loss = self.train_single_step(data)
                # print (colored(f'ERM Loss: {erm_loss.item()} Reg Loss: {reg_loss.item()}','red','on_white'))
                loss = erm_loss + self.penalty*svm_pred_loss
                loss.backward()
                self.optimizer.step()
                total_losses += loss.item()
                total_svmPred_loss += svm_pred_loss.item() if torch.is_tensor(svm_pred_loss) else svm_pred_loss
                steps +=1
            
            
            print (colored(f'Epoch {e} total Loss: {total_losses/steps}, svmLogits Pred loss: {total_svmPred_loss/steps}','red','on_white'))
            train_metric_score = self.evaluate_model(train_loader,'train')
            val_metric_score = self.evaluate_model(valid_dloader,'valid')
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            
            if test_dloader is not None:
                test_metric_score= self.evaluate_model(test_dloader,'test')
                self.test_metrics.append(test_metric_score)
                if self.delta_acc_arr is not None:
                    self.valid_test_delta_acc_list.append((val_metric_score,self.delta_acc_arr))
                
                
            self.valid_metric_list.append((val_metric_score,test_metric_score))
            # print metrics in all stages
            print (colored(f'Epoch: {e}: Train Metric: {train_metric_score} Val Metric: {val_metric_score} Test Metric: {test_metric_score}','red','on_white'))


    def evaluate_model(self, data_loader, phase):
            self.eval()
            logits_list = []
            labels_list = []
            logits_list_after_removal = []
            with torch.no_grad():
                for data in data_loader:
                    data = data.to(self.device)
                    g = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, return_both_rep=False)
                    logits = self.cls_header(g)
                    logits_list.append(logits)
                    labels_list.append(data.y)
            
            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            if self.metric_name == 'acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            elif self.metric_name == 'auc':
                metric_score = self.metric_func(all_logits[:, 1], all_labels).item()

            if phase.lower() == 'valid':
                if metric_score > self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement = 0
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement += 1
                    if self.epochs_since_improvement >= self.early_stop_epochs:
                        self.stop_training = True
            self.train()
            return metric_score
    



    def cross_entropy_with_soft_targets(self,logits, targets,y):
        """
        Calculates the cross entropy loss with soft targets manually.

        Parameters:
        logits (torch.Tensor): Logits of shape (N, C), where C is the number of classes.
        targets (torch.Tensor): Soft targets of shape (N, C), where C is the number of classes.
        targets (torch.Tensor): label targets of shape (N, )
        Returns:
        torch.Tensor: Cross entropy loss 
        """
        weights = self.reweight_logits(logits,y,gamma=self.gamma)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1)  # Sum over the class dimension
        return torch.mean(loss*weights)
    
    def reweight_logits(self,logits, y, gamma=0.5):
        """
        Reweight logits based on the given reweighting function.
        
        Parameters:
        logits (torch.Tensor): Logits tensor of shape (N, C)
        targets (torch.Tensor): Targets tensor of shape (N,)
        gamma (float): Gamma value for the reweighting function. Default is 0.5.
        
        Returns:
        torch.Tensor: Weights tensor of shape (N,)
        """
        
        # Extract the logits corresponding to the target labels
        logits = F.softmax(logits,dim=1)
        sample_logits = logits[torch.arange(logits.size(0)), y]
        
        
        # Apply the reweighting function
        weights = (1. - sample_logits ** gamma) / gamma
        
        return weights