import pickle
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


def calculate_outcomes(prior_preds, post_preds, labels, feature_index):
    """
    Parameters
    ----------
    prior_preds : np.ndarray
        Shape (n_samples, n_features). Prior model probabilities for each sample.
    post_preds : np.ndarray
        Shape (n_samples, n_features). Post model probabilities for each sample.
    labels : np.ndarray
        Shape (n_samples,). Ground truth labels (0 or 1).
        (Not used in the splitting logic below, but included per your description.)
    feature_index : int
        The index of the feature to split/group on.

    Returns
    -------
    outcome_counts : np.ndarray
        Shape (n_features, 2).  The first column is the difference vector for
        the group with `prior_preds[:, feature_index] >= 0.5`, and the second
        column is for the group with `prior_preds[:, feature_index] < 0.5`.
    """
    prior_preds = np.array(prior_preds)
    post_preds = np.array(post_preds)
    labels = np.array(labels)
    # 1) Split samples based on prior probability ≥ 0.5 or < 0.5 for feature_index
    group1_mask = prior_preds[:, feature_index] >= 0.5
    group2_mask = ~group1_mask  # same as prior_preds[:, feature_index] < 0.5

    group1_prior = prior_preds[group1_mask]
    group1_post = post_preds[group1_mask]

    group2_prior = prior_preds[group2_mask]
    group2_post = post_preds[group2_mask]

    # Helper to compute difference vector for a given group's prior/post
    def compute_difference_vector(group_prior, group_post, feat_idx):
        if len(group_prior) == 0:
            # If no samples in this group, return a zero-vector
            return np.zeros(group_prior.shape[1]) if group_prior.ndim > 1 else np.array([0.0])

        # 2) Sort by post_preds for the feature of interest.  We only need the median,
        #    so we can simply find that rather than physically sorting all samples.

        median_val = np.median(group_post[:, feat_idx])

        # 3) Split into top half vs. bottom half
        top_mask = group_post[:, feat_idx] >= median_val
        bot_mask = group_post[:, feat_idx] < median_val

        top_half = group_prior[top_mask]
        bot_half = group_prior[bot_mask]

        if len(top_half) == 0 or len(bot_half) == 0:
            # Degenerate case if all points are above or below the median
            # fallback to zero difference in that scenario
            return np.zeros(group_prior.shape[1])

        # 4) Compute averages of the prior predictions for each half
        top_avg = top_half.mean(axis=0)
        bot_avg = bot_half.mean(axis=0)

        # 5) The difference is top_avg - bot_avg
        diff_vec = top_avg - bot_avg
        return diff_vec

    # Compute difference vectors for the two groups
    diff_group1 = compute_difference_vector(group1_prior, group1_post, feature_index)
    diff_group2 = compute_difference_vector(group2_prior, group2_post, feature_index)

    # 6) Return in shape (n_features, 2) as requested
    outcome_counts = np.vstack([diff_group1, diff_group2]).T
    return outcome_counts

def sigmoid_with_epsilon(x, epsilon=1e-5):
    s = torch.sigmoid(x)
    return torch.clamp(s, min=epsilon, max=1-epsilon)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def geometric_mean(input_tensor,weights):
    log_input = torch.log(input_tensor)
    mean_log = torch.sum(log_input*weights)/torch.sum(weights)#torch.mean(log_input)
    return torch.exp(mean_log)

def unweighted_geometric_mean(input_tensor):
    log_input = torch.log(input_tensor)
    mean_log = torch.mean(log_input)
    return torch.exp(mean_log)
def unweighted_mean(input_tensor):
    return torch.mean(input_tensor)
    #return torch.sum(input_tensor*weights)/torch.sum(weights)
def weighted_mean(input_tensor,weights):
    return torch.sum(input_tensor*weights)/torch.sum(weights)

def transform_keys(keys_list, dictionary, target, number_mode=10):
    """
    For each index i in [0 .. len(keys_list)-1]:
      1. Retrieve the key = keys_list[i].
      2. Randomly pick 5 candidates from dictionary[key].
      3. Compute the distance from each candidate to target[i].
      4. Choose the candidate with the minimum distance.
      5. Collect those candidates into a PyTorch tensor and return it.
    """

    selected_elements = []

    # Ensure target has the same length as keys_list
    # so that target[i] aligns with keys_list[i].
    assert len(target) == len(keys_list), \
        "Length of target list must match length of keys_list."

    for i, key in enumerate(keys_list):
        # Access the list of candidate vectors for this key
        value_list = dictionary[key]

        # Retrieve the corresponding target vector 
        target_vector = target[i]

        # Randomly sample 5 candidates 
        # (ensure value_list has at least 5 elements)
        candidates = random.choices(value_list, k=number_mode)

        # Track the best candidate (lowest distance)
        best_candidate = None
        best_distance = float('inf')

        for c in candidates:
            # Example with Hamming distance for binary vectors
            dist = np.count_nonzero(np.array(c) != np.array(target_vector))

            # If you prefer Euclidean distance:
            # dist = np.linalg.norm(np.array(c) - np.array(target_vector))

            if dist < best_distance:
                best_distance = dist
                best_candidate = c

        selected_elements.append(best_candidate)

    # Convert selected elements to a PyTorch tensor
    selected_elements = np.array(selected_elements)
    output_tensor = torch.tensor(selected_elements, dtype=torch.float32)

    return output_tensor


def modify_tensor_based_on_shared_elements(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    # If x or y is a single vector (1D), unsqueeze to 2D
    single_vector = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        single_vector = True

    # Calculate the proportion of shared elements for each row
    # We clamp to avoid exact 1 if that’s a desired constraint. 
    # If not, you can remove the clamp logic.
    proportions_shared = torch.mean((x == y).float(), dim=1)
    # Modify each row in tensor x based on the corresponding proportion
    x_modified = x * proportions_shared[:, None] + (1 - x) * (1 - proportions_shared[:, None])

    # If originally a single vector, squeeze back
    if single_vector:
        x_modified = x_modified.squeeze(0)
        proportions_shared=proportions_shared.squeeze(0)

    return x_modified, proportions_shared

def fixed_modified(x,y):
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

    # If x or y is a single vector (1D), unsqueeze to 2D
    single_vector = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        single_vector = True

    # Calculate the proportion of shared elements for each row
    # We clamp to avoid exact 1 if that?~@~Ys a desired constraint.
    # If not, you can remove the clamp logic.
    proportions_shared = 5548/5598
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
        steps=int(torch.round(5598*(1-steps_tot[row_idx])))
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
    #print('howdy')
    #for yot in range(len(x[0])):
     #   if ((x[0][yot]>0.5) != (x_orig[0][yot])): 
      #      print('new')
       #     print(x[0][yot])
        #    print('old')
         #   print(x_orig[0][yot])
         #   print('pop')
         #   print(y_base[yot])
   # print('done')
  #  quit()
    if single_vector:
        x = x.squeeze(0)
    return x


def modify_tensor_population(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    For each vector in x (or for x itself if it is a single vector),
    compare element-wise:
       if max(y[i], 1 - y[i]) > max(x[i], 1 - x[i]),
    then replace x[i] with y[i].
    
    Args:
        x: A tensor of shape (N, D) or (D,).
        y: A tensor of shape (D,). 
            (Single vector that we compare against each vector in x.)
    
    Returns:
        A modified version of x, with the same shape it had originally.
    """

    single_vector = False
    # If x is a single vector, unsqueeze to treat it as a batch of size 1
    if x.dim() == 1:
        x = x.unsqueeze(0)
        single_vector = True

    # x shape is now (N, D), y shape is (D,)

    # Compute "confidence" for x and y: max(z, 1-z)
    x_conf = torch.max(x, 1 - x)  # shape (N, D)
    y_conf = torch.max(y, 1 - y)  # shape (D,)

    # Broadcast y_conf to match x_conf's shape for comparison
    y_conf_expanded = y_conf.unsqueeze(0).expand_as(x_conf)  # shape (N, D)

    # Create boolean mask: where y's confidence is greater than x's confidence
    mask = y_conf_expanded > x_conf

    # We will replace x[i] with y[i] wherever mask is True
    # Expand y to match x's shape so we can index properly
    y_expanded = y.unsqueeze(0).expand_as(x)

    # In-place modification where mask is True
    x[mask] = y_expanded[mask]

    # If we had to unsqueeze x, squeeze it back
    if single_vector:
        x = x.squeeze(0)

    return x

def randomly_select_vectors(binary_vectors: torch.Tensor, index_lists) -> torch.Tensor:
    """
    Given:
      binary_vectors: A tensor containing binary vectors, shape (N, d).
      index_lists: 
         - Either a single list of indices,
         - Or a list of lists, where each inner list has indices to choose from.

    This function:
      1. Randomly selects one index from each sub-list in index_lists.
      2. Returns a single chosen vector if only one list of indices was given,
         otherwise returns a tensor (M, d) where M is the number of sub-lists.
    """
    selected_vectors = []
    if isinstance(index_lists[0],np.int64):
        index_lists=[index_lists]
    for indices in index_lists:
        chosen_index = random.choice(indices)
        selected_vectors.append(binary_vectors[chosen_index])

    # If only one set of indices was given, return a single vector
    if len(selected_vectors) == 1:
        return selected_vectors[0]  # shape (d,)
    else:
        # Otherwise, stack them to get (M, d)
        return torch.stack(selected_vectors)

def proportion_mismatch_above_0_5(a, b):
    """
    Given two NumPy arrays or tensors of the same shape, this function returns
    the proportion of entries where 'a' is above 0.5 and 'b' is below 0.5, or
    'a' is below 0.5 and 'b' is above 0.5.
    
    Parameters:
    -----------
    a : np.ndarray
        A NumPy array (or a similar array-like structure).
    b : np.ndarray
        A NumPy array of the same shape as 'a'.
        
    Returns:
    --------
    float
        The proportion of entries for which (a>0.5) != (b>0.5).
    """
    # Threshold both a and b at 0.5, producing boolean arrays
    a_above = (a > 0.5)
    b_above = (b > 0.5)
    
    # Compute mismatches (i.e., XOR, or != in boolean context)
    mismatches = a_above != b_above
    
    # Return the proportion of mismatches
    return mismatches.float().mean()

def count_appearences(x,y,z):
    index_of_choice=2239
    number_of_appearences=0
    relative_count=0
    initial=z[index_of_choice]
    for entry in x:
        if torch.sum(entry!=z)<=100:
            relative_count=relative_count+1
            if entry[index_of_choice]!=z[index_of_choice]:
                number_of_appearences=number_of_appearences+1
    if relative_count>10:
        final_value=number_of_appearences/relative_count
    else:
        relative_count=10
        number_of_appearences=0
        for entry in y:
            if x[entry][index_of_choice]!=z[index_of_choice]:
                number_of_appearences=number_of_appearences+1
        final_value=number_of_appearences/relative_count
    return final_value, initial
# A helper function that checks one entry

class valDataset(Dataset):
    def __init__(self, samples, num_samples, baselines, weights,ST_labels,train_ref,indices,pop_mean):
        super(valDataset, self).__init__()
        self.num_samples =num_samples

        self.labels = torch.tensor(samples, dtype=torch.float32)
        self.baselines = torch.tensor(baselines, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.STs=ST_labels
        self.train_ref=torch.tensor(train_ref,dtype=torch.float32)
        self.indices=indices
        self.pop_mean=torch.tensor(pop_mean,dtype=torch.float32)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Apply noise on the fly using tensor operations
        number_changed,initial_state=count_appearences(self.train_ref,self.indices[idx],self.labels[idx])
        noised_samples,number_shared=fixed_modified(self.labels[idx],self.labels[idx])
        number_shared=torch.tensor(number_shared,dtype=torch.float32)
        old_sample=noised_samples.clone()
        pop_modified=modify_advanced(noised_samples,self.pop_mean,number_shared)
        #print(proportion_mismatch_above_0_5(old_sample,pop_modified))
        return pop_modified, number_changed,initial_state  # Remove batch dimension after noise application

class SampleDataset(Dataset):
    def __init__(self, samples, num_samples, baselines, weights,ST_labels,train_ref,indices,pop_mean):
        super(SampleDataset, self).__init__()
        self.num_samples =num_samples
        self.labels = torch.tensor(samples, dtype=torch.float32)
        self.baselines = torch.tensor(baselines, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.STs=ST_labels
        self.train_ref=torch.tensor(train_ref,dtype=torch.float32)
        self.indices=indices
        self.pop_mean=torch.tensor(pop_mean,dtype=torch.float32)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Apply noise on the fly using tensor operations
        to_obtain=randomly_select_vectors(self.train_ref,self.indices[idx])
        noised_samples,number_shared=modify_tensor_based_on_shared_elements(self.labels[idx],to_obtain)
        old_sample=noised_samples.clone()
        pop_modified=modify_advanced(noised_samples,self.pop_mean,number_shared)
        #pop_modified=modify_tensor_population(noised_samples,self.pop_mean) 
        #print(proportion_mismatch_above_0_5(old_sample,pop_modified))
        return pop_modified, to_obtain,self.baselines[idx],self.weights[idx],old_sample  # Remove batch dimension after noise application

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
        

        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        identity = x
        
        # Optional: Tanh on the input (as in your code)
        out = F.tanh(x)
        
        # First layer + BatchNorm + activation
        #out=self.dropout(out)
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
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        identity = x
        out=F.tanh(x)
        #out=self.dropout(out)
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


def focal_loss(inputs, targets, gamma):
    ''' Focal loss function for binary classification '''
    # BCE loss with logits to handle the numerical stability
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

    # Calculating p_t
    pt = torch.where(targets == 1, inputs, 1 - inputs)

    # Calculate the modulating factor (1 - p_t)^gamma
    modulating_factor = (1 - pt) ** gamma

    # Calculate final focal loss
    focal_loss = modulating_factor * bce_loss

    return focal_loss

def perform_validation(model, val_dataloader, baseline, pop_mean,validation_picks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    big_list=[]
    big_list2=[]
    phylo_correctness=[]
    phylo_correctness2=[]
    model_correctness=[]
    with torch.no_grad():
        for batch_idx, (data, target, baseline, weights,guess) in enumerate(val_dataloader):
            data, target, baseline, weights,guess = data.to(device), target.to(device), baseline.to(device),weights.to(device),guess.to(device)
            output = model(data.float())
            loss = focal_loss(output, target.float(), 0)
            correct_predictions = ((output > 0.5) == target.byte()).float()
            model_correctness.append(weighted_mean(correct_predictions.mean(dim=1),weights).detach().cpu())
            used_base = torch.logit(guess.float(), eps=1e-5)
            howdy=torch.logit(data.float(),eps=1e-5)
            prior = sigmoid_with_epsilon(used_base)
            prior2 = sigmoid_with_epsilon(howdy)
            base_loss = focal_loss(prior, target.float(), 0)
            base_loss2 = focal_loss(prior2, target.float(), 0)
            correct_predictions = ((prior > 0.5) == target.byte()).float()
            correct_predictions2 = ((prior2 > 0.5) == target.byte()).float()
            phylo_correctness.append(weighted_mean(correct_predictions.mean(dim=1),weights).detach().cpu())
            phylo_correctness2.append(weighted_mean(correct_predictions2.mean(dim=1),weights).detach().cpu())
           # phylo_correctness2.append(correct_predictions2.mean().detach().cpu())

            # Normalize losses relative to sample
            sample_losses = base_loss.sum(dim=1).detach()
            sample_losses2 = base_loss2.sum(dim=1).detach()
           # normalized_loss = (loss / sample_losses[:, None])#*(weights.detach()[:, None])
            batch_loss=geometric_mean(loss.sum(dim=1).detach(),weights)/geometric_mean(sample_losses,weights)
            batch_loss2=geometric_mean(loss.sum(dim=1).detach(),weights)/geometric_mean(sample_losses2,weights)
            #normalized_loss2 = (loss / sample_losses2[:, None])#*(weights.detach()[:, None])
            # Sum up all the normalized losses for the entire batch
           # batch_loss = geometric_mean(normalized_loss.sum(dim=1))#normalized_loss.sum() / len(sample_losses)
            #batch_loss2 = geometric_mean(normalized_loss2.sum(dim=1),weights)#normalized_loss.sum() / len(sample_losses)
            big_list.append((1/batch_loss.detach()).cpu())
            big_list2.append((1/batch_loss2.detach()).cpu())
    print('base proportion')
    print(np.median(phylo_correctness))
    print('ST proportion')
    print(np.median(phylo_correctness2))
    print('model proportion')
    print(np.median(model_correctness))
    print('start')
    print(np.median(big_list))
    print('advanced')

    return (np.median(big_list2))

def perform_test(model, val_dataloader,baseline,pop_mean,validation_picks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    big_list2=[]
    with torch.no_grad():
        for batch_idx, (data, target, baseline, weights,guess) in enumerate(val_dataloader):
            data, target, baseline, weights,guess = data.to(device), target.to(device), baseline.to(device),weights.to(device),guess.to(device)
            #data=modify_tensor_based_on_shared_elements(second_data,target)
            log_data=torch.logit(data.float(),eps=1e-5)
            output = model(data.float())
            base_output=sigmoid_with_epsilon(log_data,epsilon=1e-5)
            pop_output=sigmoid_with_epsilon(torch.logit(torch.tensor(pop_mean).to(device).float(),eps=1e-5),epsilon=1e-5)
            st_output=sigmoid_with_epsilon(torch.logit(baseline.float(),eps=1e-5),epsilon=1e-5)
            model_predict=calculate_outcomes(guess.float().cpu(),output.cpu(),target.cpu(),2239)
            pop_predict=calculate_outcomes(guess.float().cpu(),target.cpu(),target.cpu(),2239)
            st_predict=calculate_outcomes(guess.float().cpu(),st_output.cpu(),target.cpu(),2239)
            base_predict=calculate_outcomes(guess.float().cpu(),base_output.cpu(),target.cpu(),2239)
            big_predict=np.concatenate((model_predict,pop_predict,st_predict,base_predict),axis=1)
            print(big_predict)
    return (big_predict)


def big_test(model, test_dataloader, output_csv='output.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Optionally, write a header:
        writer.writerow(['data_2239', 'number_of_things', 'initial_state'])

        with torch.no_grad():
            # Remove the single-batch assumption:
            for data, number_of_things, initial_state in test_dataloader:
                # Move inputs to the appropriate device
                data = data.to(device).float()
                number_of_things = number_of_things.to(device)
                initial_state = initial_state.to(device)

                # Forward pass through the model
                outputs = model(data)
                
                # For each item in the batch, save the selected values
                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    value_data_2239 = outputs[i, 2239].item()
                    value_number_of_things = number_of_things[i].item()
                    value_initial_state = initial_state[i].item()

                    # Write a row per sample
                    writer.writerow([
                        value_data_2239,
                        value_number_of_things,
                        value_initial_state
                    ])

    print(f"Finished writing data to {output_csv}")
    torch.save(model.state_dict(), 'twolayer.pth')
    quit()
    return(None)

def train_model(model_config, learning_rate, weight_decay, gamma, train_dataset,val_dataset,test_dataset,pop_mean,num_epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders with multiple workers
    num_workers = cpu_count()  # Use all available CPU cores
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=20)
    
    # Instantiate the model and wrap it with DataParallel
    model = CustomResNet(num_features=5598, config=model_config)
    model = DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_score = 0
    no_improvement_count = 0
    l1_lambda=1e-6
    l2_lambda=1e-5
    performance_watch=[]
    for epoch in range(num_epochs):
        model.train()
        print(epoch)
        big_list=[]
        loss_ratios_l1=[]
        loss_ratios_l2=[]
        for batch_idx, (data, target, baseline, weights,guess) in enumerate(train_dataloader):
            data, target, baseline, weights,guess = data.to(device), target.to(device), baseline.to(device),weights.to(device),guess.to(device)

            optimizer.zero_grad()
            output = model(data.float())
            used_base = torch.logit(data.float(), eps=1e-5)
            prior = sigmoid_with_epsilon(used_base)
            base_loss = focal_loss(prior, target.float(), gamma) #added gamma back in from 0
            loss=focal_loss(output,target.float(),gamma)

            # Normalize losses relative to sample
            sample_losses = base_loss.sum(dim=1).detach()
            normalized_loss = loss / sample_losses[:, None]#*(weights.detach()[:,None])
            # Sum up all the normalized losses for the entire batch
            batch_loss = normalized_loss.sum()/len(sample_losses)# / weights.detach()[:,None].sum()
            
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.square().sum() for p in model.parameters())
            total_loss = batch_loss + l2_lambda*l2_norm+l1_norm*l1_lambda
            loss_ratios_l1.append(batch_loss.detach().cpu()/(l1_lambda * l1_norm).detach().cpu())
            loss_ratios_l2.append(batch_loss.detach().cpu()/(l2_lambda * l2_norm).detach().cpu())

            #batch_loss = loss.sum()/len(sample_losses)# / weights.detach()[:,None].sum()
            big_list.append((1/batch_loss.detach()).cpu())
            total_loss.backward()
            #batch_loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
        print(np.median(big_list))
        print('loss ratio l1')
        print(np.median(loss_ratios_l1))
        print('loss ratio l2')
        print(np.median(loss_ratios_l2))
        # Validate the model at some interval or at the end of each epoch
        if epoch%1==0 or epoch == num_epochs - 1:
            val_score = perform_validation(model, val_dataloader, baseline.float(),pop_mean,validation_picks)
            performance_watch.append(val_score)
            # Early stopping check
            if val_score > best_val_score:
                best_val_score = val_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if len(performance_watch)>20:
                print('difference')
                print(np.mean(performance_watch[-10:])-np.mean(performance_watch[-20:-10]))
                if np.mean(performance_watch[-10:])-np.mean(performance_watch[-20:-10])<=0 or epoch>=500:
                    print(f"Stopping early at epoch {epoch}: Validation score did not improve.")
                    break

            print(f"Epoch {epoch}: Validation Score = {val_score}")
    end_output=[]
    big_test(model,test_dataloader)
    quit()
    for i in range(10):
        end_output.append(perform_test(model, val_dataloader,baseline.float(),pop_mean, validation_picks))

    return np.mean(end_output,axis=0)


def cross_validate_model(data, base_data, pop_mean, weights, dates,validation_picks,labels,model_config, learning_rate, weight_decay, gamma, num_epochs=500):
    # Ensure the date column is in datetime format
    dates = pd.to_datetime(dates)
    
# Split data based on the year using boolean masks
    train_mask = dates.year <= 2019
    val_mask1 = dates.year == 2019
    val_mask2 = dates.year == 2020

    # Convert boolean masks to integer indices for numpy array indexing
    train_indices = np.where(train_mask)[0]
    val_indices1 = np.where(val_mask1)[0]
    val_indices2 = np.where(val_mask2)[0]
    # Use indices to select samples from numpy arrays
    train_samples = data[train_indices]
    val_samples1 = data[val_indices1]
    val_samples2 = data[val_indices2]

    train_baseline = base_data[train_indices]
    val_baseline1 = base_data[val_indices1]
    val_baseline2=base_data[val_indices2]
    train_weights = weights[train_indices]
    val_weights1 = weights[val_indices1]
    val_weights2 = weights[val_indices2]

    train_labels=labels[train_indices]
    val_labels1=labels[val_indices1]
    val_labels2=labels[val_indices2]

    all_train=np.loadtxt('ten_indices.txt').astype(int)
    all_val=np.loadtxt('val_ten_indices.txt').astype(int)
    new_train=[]
    new_baseline=[]
    new_weights=[]
    new_labels=[]
    used_indices=[]
    for i in range(len(all_train)):
        current_indices=all_train[i]
        if current_indices[0]!=-1:
            new_train.append(train_samples[i])
            new_baseline.append(train_baseline[i])
            new_weights.append(train_weights[i])
            new_labels.append(train_labels[i])
            used_indices.append(current_indices)
    
    
    train_dataset = SampleDataset(np.array(new_train), len(new_train), np.array(new_baseline),np.array(new_weights),new_labels,np.array(train_samples),used_indices,pop_mean)
    val_dataset = SampleDataset(val_samples1, len(val_samples1), val_baseline1,val_weights1,val_labels1,np.array(val_samples2),all_val,pop_mean)
    test_dataset = valDataset(val_samples1, len(val_samples1), val_baseline1,val_weights1,val_labels1,np.array(val_samples2),all_val,pop_mean)
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model on training data and validate on validation data
    val_score = train_model(model_config, learning_rate, weight_decay, gamma, train_dataset,val_dataset,test_dataset,pop_mean,num_epochs)
    #print(f"Validation score: {val_score}")

    # Optionally test model on test data here, similar to validation
    np.savetxt("mecA.csv", val_score,delimiter=',')
    return val_score

def process_file(filename: str, folder_path: str) -> Tuple[str, np.ndarray]:
    sequence_type = filename.split("__")[-3].split(".")[0]
    date= filename.split("__")[-1].split(".")[0]
    if date=='0000':
        date=str(int(filename.split("__")[-2].split(".")[0])-3)
    with open(os.path.join(folder_path, filename), 'r') as file:
        lines = file.readlines()
        binary_vector = np.array([int(line.strip().split('\t')[0]) for line in lines if line.strip()])
    return sequence_type, binary_vector, date

def prepare_dataset_and_averages(folder_path: str, min_members: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    starting_filenames = sorted(
    f for f in os.listdir(folder_path) if f.endswith(".txt"))
    filenames=[]
    # Count the number of files for each sequence type
    sequence_type_counts = {}
    for filename in starting_filenames:
        seq_type = filename.split("__")[-3].split(".")[0]
        collection_date=filename.split("__")[-1].split(".")[0]
        upload_date=filename.split("__")[-2].split(".")[0]
        if collection_date != '0000' or int(upload_date)<2020:
            sequence_type_counts[seq_type] = sequence_type_counts.get(seq_type, 0) + 1
            filenames.append(filename)
    # Determine which sequence types have enough members
    valid_sequence_types = {seq_type for seq_type, count in sequence_type_counts.items() if count >= min_members}
    # Use all available CPUs
    num_cpus = cpu_count()

    # Create a partial function with the folder_path argument filled in
    process_file_partial = partial(process_file, folder_path=folder_path)

    # Use multiprocessing to process files in parallel
    with Pool(num_cpus) as pool:
        results = pool.map(process_file_partial, filenames)

    # Prepare the dataset and calculate averages
    dataset = []
    labels = []
    date_list=[]
    sequence_type_data = defaultdict(lambda: {'sum': None, 'count': 0})
    label_keys={}
    label_indices=[]
    index=0
    found=0

    for sequence_type, binary_vector, date in results:
        if sequence_type in valid_sequence_types:
            dataset.append(binary_vector)
            labels.append(sequence_type)
            date_list.append(date)
            target_type = sequence_type
        else:
            dataset.append(binary_vector)
            date_list.append(date)
            labels.append('other')
            target_type = 'other'

        if sequence_type_data[target_type]['sum'] is None:
            label_keys[target_type]=found
            label_indices.append([])
            found=found+1
            sequence_type_data[target_type]['sum'] = np.zeros_like(binary_vector)

        sequence_type_data[target_type]['sum'] += binary_vector
        sequence_type_data[target_type]['count'] += 1
        label_indices[label_keys[target_type]].append(index)
        index=index+1
    
    # Calculate average frequencies
    average_frequencies = {}
    weights={}
    for sequence_type, data in sequence_type_data.items():
        average_frequencies[sequence_type] = data['sum'] / data['count']
        weights[sequence_type]=data['count']
    
    train_freqs=defaultdict(lambda: {'sum': None, 'count': 0})
    validation_picks={}
    for i in range(len(dataset)):
        ST=labels[i]
        binary_vector=dataset[i]
        current_date=date_list[i]
        if int(current_date)<2020:
            if train_freqs[ST]['sum'] is None:
                train_freqs[ST]['sum']=np.zeros_like(binary_vector)
                validation_picks[ST]=[]
            train_freqs[ST]['sum'] += binary_vector
            validation_picks[ST].append(binary_vector)
            train_freqs[ST]['count'] += 1
    validation_frequencies={}
    for sequence_type, data in train_freqs.items():
        validation_frequencies[sequence_type] = data['sum'] / data['count']
        weights[sequence_type]=data['count']
    population_mean=np.array(list(validation_frequencies.values())).mean(axis=0)
    # Create an array of average frequencies corresponding to each sample
    avg_freq_array = np.array(
    [validation_frequencies[label] if label in validation_frequencies else population_mean
     for label in labels])
    avg_weights_array = np.array([1 for label in labels])
    true_data_set=np.array(dataset)
    return true_data_set, avg_freq_array, population_mean, avg_weights_array, date_list, labels, validation_picks

print('hello world')
data, base_data, pop_mean, weights, dates, labels, validation_picks=prepare_dataset_and_averages('all_embeded_genomes')
print(len(data))
feature_size=100
model_config = [{'type': 'linear', 'intermediate_features': 5598},
            {'type': 'leaky', 'intermediate_features': 5598, 'num_intermediate_layers': 0}]


cross_validate_model(data, base_data, pop_mean,weights,dates, validation_picks,np.array(labels),model_config, 0.00001, 0,0)
