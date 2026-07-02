import csv
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Dict
from collections import defaultdict
from torch.nn.parallel import DataParallel
import random
import argparse

N_FEATURES = 5598

def sigmoid_with_epsilon(x, epsilon=1e-5):
    s = torch.sigmoid(x)
    return torch.clamp(s, min=epsilon, max=1-epsilon)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def geometric_mean(input_tensor,weights):
    log_input = torch.log(input_tensor)
    mean_log = torch.sum(log_input*weights)/torch.sum(weights)
    return torch.exp(mean_log)

def unweighted_geometric_mean(input_tensor):
    log_input = torch.log(input_tensor)
    mean_log = torch.mean(log_input)
    return torch.exp(mean_log)
def unweighted_mean(input_tensor):
    return torch.mean(input_tensor)
def weighted_mean(input_tensor,weights):
    return torch.sum(input_tensor*weights)/torch.sum(weights)

def modify_tensor_based_on_shared_elements(x, num_not_shared: int, n_total: int = N_FEATURES):
    """
    Modify each row in tensor x based on the proportion of shared elements
    with corresponding rows in tensor y.

    If x,y are 1D, treat them as single rows and return a 1D result.

    Parameters:
      x (torch.Tensor): 1D or 2D tensor containing 0s and 1s.
      y (torch.Tensor): 1D or 2D tensor containing 0s and 1s; must be same size as x.

    Returns:
      torch.Tensor: Modified version of tensor x (same shape as input).
    """
    x=torch.tensor(x,dtype=torch.float32)
    # If x or y is a single vector (1D), unsqueeze to 2D
    single_vector = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        single_vector = True

    # Calculate the proportion of shared elements for each row
    # We clamp to avoid exact 1 if that?~@~Ys a desired constraint.
    # If not, you can remove the clamp logic.
    proportions_shared = (n_total - num_not_shared) / n_total
    proportions_shared=torch.tensor(proportions_shared,dtype=torch.float32)
    # Modify each row in tensor x based on the corresponding proportion
    x_modified = x * proportions_shared + (1 - x) * (1 - proportions_shared)

    # If originally a single vector, squeeze back
    if single_vector:
        x_modified = x_modified.squeeze(0)

    return x_modified, proportions_shared

def modify_advanced(x, y,steps_tot):
    
    single_vector = False
    # If x is a single vector, unsqueeze to treat it as a batch of size 1
    if x.dim() == 1:
        x = x.unsqueeze(0)
        steps_tot=steps_tot.unsqueeze(0)
        single_vector = True

    # x is now (N, D), y is (D,)
    N, D = x.shape
    x_orig=x.clone()
    y_base = y.clone()

    for row_idx in range(N):
        row = x[row_idx]
        steps=int(torch.round(N_FEATURES*(1-steps_tot[row_idx])))
        modified_y = y_base.clone()
        mask_le = (row <= 0.5)
        mask_gt = ~mask_le  # row > 0.5

        modified_y[mask_le] = y_base[mask_le] / (1 - y_base[mask_le])
        modified_y[mask_gt] = (1 - y_base[mask_gt]) / y_base[mask_gt]
       
        modified_y=torch.sqrt(modified_y)
        modified_y = torch.clamp(modified_y, max=10_000)
        final_vector = torch.zeros_like(modified_y)

        excluded_count = 0
        big_sum=modified_y.sum()
        counter=0
        
        final_vector=steps*(modified_y/big_sum)
        
        mask_overflow=(final_vector>1)
        mask_underflow=(final_vector<1)
        while (mask_overflow.any()) and counter<10:
            leftover=(final_vector[mask_overflow]-1).sum()
            final_vector[mask_overflow]=1
            new_sum=modified_y[mask_underflow].sum()
            final_vector[mask_underflow]=leftover*(modified_y[mask_underflow])/new_sum+final_vector[mask_underflow]
            mask_overflow=(final_vector>1)
            mask_underflow=(final_vector<1)
            counter=counter+1
        
        final_vector[mask_gt] = 1.0 - final_vector[mask_gt]        
        final_vector = torch.clamp(final_vector, 1e-5, 1 - 1e-5)
        x[row_idx] = final_vector
    if single_vector:
        x = x.squeeze(0)
    return x

class SampleDataset(Dataset):
    def __init__(self, samples, num_samples, pop_mean, num_not_shared: int, n_total: int = N_FEATURES):
        super(SampleDataset, self).__init__()
        self.num_samples = num_samples
        self.labels = torch.tensor(samples, dtype=torch.float32)
        self.pop_mean = torch.tensor(pop_mean, dtype=torch.float32)
        self.num_not_shared = int(num_not_shared)  
        self.n_total = int(n_total)                
    
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Apply noise on the fly using tensor operations
        noised_samples, number_shared = modify_tensor_based_on_shared_elements(
            self.labels[idx],
            num_not_shared=self.num_not_shared,
            n_total=self.n_total,
        )
        pop_modified=modify_advanced(noised_samples,self.pop_mean,number_shared)
        return pop_modified  # Remove batch dimension after noise application

class ResidualBlock(nn.Module):
    def __init__(self, num_features, intermediate_features=200, num_intermediate_layers=0):
        super(ResidualBlock, self).__init__()
        
        # First linear + batchnorm
        self.fc1 = nn.Linear(num_features, intermediate_features)
        self.bn1 = nn.BatchNorm1d(intermediate_features)  # Match intermediate_features, not num_features
        
        # Create pairs of (Linear, BatchNorm) for each intermediate layer
        self.intermediate_layers = nn.ModuleList([
            nn.Linear(intermediate_features, intermediate_features)
            for _ in range(num_intermediate_layers)
        ])
        self.intermediate_bns = nn.ModuleList([
            nn.BatchNorm1d(intermediate_features)
            for _ in range(num_intermediate_layers)
        ])
        
        
        self.fc2 = nn.Linear(intermediate_features,intermediate_features)
        # Final linear
        self.fc3 = nn.Linear(intermediate_features, num_features)
        
    def forward(self, x):
        identity = x
        
        # Optional: Tanh on the input (as in your code)
        out = F.tanh(x)
        
        # First layer + BatchNorm + activation
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        
        # Intermediate layers, each followed by BatchNorm + activation
        for layer, bn in zip(self.intermediate_layers, self.intermediate_bns):
            out = layer(out)
            out = bn(out)
            out = F.leaky_relu(out)
        
        # Final linear and residual connection
        out = self.fc3(out)
        out += identity
        
        return out

class InteractionBlock(nn.Module):
    def __init__(self, num_features, intermediate_features=200):
        super(InteractionBlock, self).__init__()
        self.fc1 = nn.Linear(num_features, intermediate_features)
        self.fc2 = nn.Linear(intermediate_features, num_features)
        self.ln1 = nn.LayerNorm(num_features)
    def forward(self, x):
        identity = x
        out=F.tanh(x)
        out = self.fc2(out)
        out += identity
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_features, config):
        super(CustomResNet, self).__init__()
        self.layers = nn.ModuleList()
        for layer_config in config:
            if layer_config['type'] == 'leaky':
                block = ResidualBlock(num_features, layer_config.get('intermediate_features', 200), layer_config.get('num_intermediate_layers', 1))
                self.layers.append(block)
            elif layer_config['type'] == 'linear':
                block = InteractionBlock(num_features, layer_config.get('intermediate_features', 200))
                self.layers.append(block)
            elif layer_config['type'] == 'token':
                block = TokenizedTransformerBlock(num_features,layer_config.get('token_size'),layer_config.get('intermediate_features'))
                self.layers.append(block)

    def forward(self, x):
        out = torch.logit(x, eps=1e-5)
        for layer in self.layers:
            out = layer(out)
        out = sigmoid_with_epsilon(out)  # Output squashed to [0, 1]
        return out


def perform_test(model, use_dataloader,pop_mean):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(use_dataloader):
            data  = data.to(device)
            log_data=torch.logit(data.float(),eps=1e-5)
            output = model(data.float()).cpu()
    return output


def run_model(input_genomes, num_not_shared: int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = N_FEATURES
    config = [
    {"type": "linear", "intermediate_features": N},             # layer 0
    {"type": "leaky",  "intermediate_features": N,
     "num_intermediate_layers": 0},                             # layer 1 (ResidualBlock)
        ]

    model = CustomResNet(N, config)
    ckpt = torch.load("twolayer.pth", map_location="cpu")
    if any(k.startswith("module.") for k in ckpt):
        ckpt = {k.removeprefix("module."): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    pop_mean=np.loadtxt('pop_mean.txt')
    # Create DataLoaders with multiple workers
    num_workers = cpu_count()  # Use all available CPU cores

    use_dataset = SampleDataset(input_genomes, len(input_genomes), pop_mean,num_not_shared)
    use_dataloader = DataLoader(use_dataset, batch_size=len(use_dataset), shuffle=False, num_workers=20)
    return(perform_test(model, use_dataloader,pop_mean))

