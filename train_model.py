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


def calculate_outcomes(prior_prob_matrix, post_prob_matrix, labels, split_feature_idx):
    """
    Compute per-feature *difference vectors* for two groups defined by a "prior"
    probability threshold on a particular feature.

    High level:
      1) Split samples into two groups depending on whether the PRIOR probability
         for split_feature_idx is >= 0.5 or < 0.5.
      2) Within each group, use the POST probabilities for split_feature_idx to
         split samples into top vs bottom halves around the median.
      3) For each half, compute the mean PRIOR probability vector across *all* features.
      4) Return (top_mean - bottom_mean) for each group, stacked as (n_features, 2).

    Parameters
    ----------
    prior_prob_matrix : np.ndarray
        Shape (n_samples, n_features). Prior model probabilities for each sample.
    post_prob_matrix : np.ndarray
        Shape (n_samples, n_features). Post model probabilities for each sample.
    labels : np.ndarray
        Shape (n_samples,). Ground truth labels (0 or 1).
        (Not used in the splitting logic below, but included for signature consistency.)
    split_feature_idx : int
        The index of the feature to split/group on.

    Returns
    -------
    outcome_diff_by_group : np.ndarray
        Shape (n_features, 2). Column 0 = diff vector for group where
        prior_prob_matrix[:, split_feature_idx] >= 0.5, column 1 = diff vector for
        group where prior_prob_matrix[:, split_feature_idx] < 0.5.
    """
    prior_prob_matrix = np.array(prior_prob_matrix)
    post_prob_matrix = np.array(post_prob_matrix)
    labels = np.array(labels)

    # 1) Split samples based on PRIOR probability threshold for split_feature_idx
    prior_high_mask = prior_prob_matrix[:, split_feature_idx] >= 0.5
    prior_low_mask = ~prior_high_mask  # same as < 0.5

    prior_high_group_prior = prior_prob_matrix[prior_high_mask]
    prior_high_group_post = post_prob_matrix[prior_high_mask]

    prior_low_group_prior = prior_prob_matrix[prior_low_mask]
    prior_low_group_post = post_prob_matrix[prior_low_mask]

    def compute_difference_vector(group_prior_probs, group_post_probs, feat_idx):
        """
        For a single group:
          - split by POST median on feat_idx into top/bottom halves,
          - compute mean PRIOR prob vector for each half,
          - return top_mean - bottom_mean.
        """
        if len(group_prior_probs) == 0:
            # No samples in this group -> zero vector (match expected n_features)
            return np.zeros(group_prior_probs.shape[1]) if group_prior_probs.ndim > 1 else np.array([0.0])

        # We only need the median of the POST probabilities for the chosen feature.
        post_median = np.median(group_post_probs[:, feat_idx])

        # 3) Split into top vs bottom halves by POST median
        top_half_mask = group_post_probs[:, feat_idx] >= post_median
        bottom_half_mask = group_post_probs[:, feat_idx] < post_median

        top_half_prior = group_prior_probs[top_half_mask]
        bottom_half_prior = group_prior_probs[bottom_half_mask]

        if len(top_half_prior) == 0 or len(bottom_half_prior) == 0:
            # Degenerate case: everyone falls on one side of the median
            return np.zeros(group_prior_probs.shape[1])

        # 4) Mean prior probabilities for each half
        top_mean_prior = top_half_prior.mean(axis=0)
        bottom_mean_prior = bottom_half_prior.mean(axis=0)

        # 5) Difference vector
        return top_mean_prior - bottom_mean_prior

    # Difference vectors for each group
    diff_prior_high = compute_difference_vector(prior_high_group_prior, prior_high_group_post, split_feature_idx)
    diff_prior_low = compute_difference_vector(prior_low_group_prior, prior_low_group_post, split_feature_idx)

    # Return as (n_features, 2)
    outcome_diff_by_group = np.vstack([diff_prior_high, diff_prior_low]).T
    return outcome_diff_by_group


def sigmoid_with_epsilon(logits_tensor, epsilon=1e-5):
    """Sigmoid, then clamp to avoid exact 0/1 (prevents logit/CE numerical issues downstream)."""
    probs = torch.sigmoid(logits_tensor)
    return torch.clamp(probs, min=epsilon, max=1 - epsilon)


def logistic(x):
    """Standard logistic for numpy arrays."""
    return 1 / (1 + np.exp(-x))


def geometric_mean(prob_tensor, weight_tensor):
    """
    Weighted geometric mean: exp( sum(w*log(p)) / sum(w) ).
    Assumes prob_tensor is strictly > 0 (caller clamps elsewhere).
    """
    log_prob = torch.log(prob_tensor)
    weighted_log_mean = torch.sum(log_prob * weight_tensor) / torch.sum(weight_tensor)
    return torch.exp(weighted_log_mean)


def unweighted_geometric_mean(prob_tensor):
    """Unweighted geometric mean over all entries."""
    log_prob = torch.log(prob_tensor)
    return torch.exp(torch.mean(log_prob))


def unweighted_mean(x_tensor):
    """Plain mean helper (kept for parity with other mean helpers)."""
    return torch.mean(x_tensor)


def weighted_mean(x_tensor, weight_tensor):
    """Weighted arithmetic mean: sum(w*x)/sum(w)."""
    return torch.sum(x_tensor * weight_tensor) / torch.sum(weight_tensor)


def transform_keys(key_list, key_to_vectors, target_vectors, number_mode=10):
    """
    For each i:
      1) key = key_list[i]
      2) sample `number_mode` candidate vectors from key_to_vectors[key]
      3) compute distance from each candidate to target_vectors[i]
      4) choose closest candidate
      5) return all chosen candidates as a float32 torch tensor

    Notes:
      - Uses Hamming distance via count_nonzero != (binary-ish vectors).
      - Uses random.choices (sampling WITH replacement).
    """
    chosen_vectors = []

    # Ensure per-row alignment: target_vectors[i] corresponds to key_list[i]
    assert len(target_vectors) == len(key_list), "Length of target_vectors must match length of key_list."

    for i, key in enumerate(key_list):
        candidate_pool = key_to_vectors[key]
        target_vec = target_vectors[i]

        # Randomly sample candidates from the pool
        sampled_candidates = random.choices(candidate_pool, k=number_mode)

        # Select the closest candidate
        best_candidate = None
        best_distance = float("inf")

        for candidate in sampled_candidates:
            # Hamming distance (count mismatched entries)
            dist = np.count_nonzero(np.array(candidate) != np.array(target_vec))

            # Alternative (commented) Euclidean distance:
            # dist = np.linalg.norm(np.array(candidate) - np.array(target_vec))

            if dist < best_distance:
                best_distance = dist
                best_candidate = candidate

        chosen_vectors.append(best_candidate)

    chosen_vectors = np.array(chosen_vectors)
    return torch.tensor(chosen_vectors, dtype=torch.float32)


def modify_tensor_based_on_shared_elements(input_vecs: torch.Tensor, reference_vecs: torch.Tensor) -> torch.Tensor:
    """
    Blend each input vector toward its complement depending on the *proportion of
    shared elements* with the corresponding reference vector.

    If input_vecs/reference_vecs are 1D, they are treated as a single row and a 1D result is returned.

    Parameters:
      input_vecs (torch.Tensor): 1D or 2D tensor containing 0s and 1s.
      reference_vecs (torch.Tensor): same shape as input_vecs, also 0s and 1s.

    Returns:
      modified_input (torch.Tensor): same shape as input_vecs
      proportions_shared (torch.Tensor): shape (N,) or scalar if 1D input
    """
    single_row = False
    if input_vecs.dim() == 1:
        input_vecs = input_vecs.unsqueeze(0)
        reference_vecs = reference_vecs.unsqueeze(0)
        single_row = True

    # Proportion of positions where input matches reference per row
    proportions_shared = torch.mean((input_vecs == reference_vecs).float(), dim=1)

    # Per-element blending using the row's shared proportion
    modified_input = (
        input_vecs * proportions_shared[:, None]
        + (1 - input_vecs) * (1 - proportions_shared[:, None])
    )

    if single_row:
        modified_input = modified_input.squeeze(0)
        proportions_shared = proportions_shared.squeeze(0)

    return modified_input, proportions_shared


def fixed_modified(input_vecs, reference_vecs):
    """
    Same interface as modify_tensor_based_on_shared_elements, but uses a fixed
    global shared proportion (5548/5598) instead of computing from reference_vecs.

    Note: reference_vecs is unused, kept only to preserve the original call signature.
    """
    single_row = False
    if input_vecs.dim() == 1:
        input_vecs = input_vecs.unsqueeze(0)
        reference_vecs = reference_vecs.unsqueeze(0)
        single_row = True

    fixed_shared_proportion = 5548 / 5598

    modified_input = (
        input_vecs * fixed_shared_proportion
        + (1 - input_vecs) * (1 - fixed_shared_proportion)
    )

    if single_row:
        modified_input = modified_input.squeeze(0)

    return modified_input, fixed_shared_proportion


def modify_advanced(input_probs, population_mean_probs, shared_proportion_per_row):
    """
    Advanced transformation that converts a probabilistic vector into a new vector
    with a row-specific number of "steps" (derived from shared_proportion_per_row),
    while respecting whether entries were originally <=0.5 or >0.5.

    IMPORTANT:
      - Kept logic identical to your original implementation.
      - Variable names + comments updated for readability only.
    """
    single_row = False
    if input_probs.dim() == 1:
        input_probs = input_probs.unsqueeze(0)
        shared_proportion_per_row = shared_proportion_per_row.unsqueeze(0)
        single_row = True

    batch_size, num_features = input_probs.shape
    original_input = input_probs.clone()
    pop_mean = population_mean_probs.clone()

    for row_idx in range(batch_size):
        row_probs = input_probs[row_idx]

        # "steps" is derived from the shared proportion with a fixed 5598 scaling
        steps = int(torch.round(5598 * (1 - shared_proportion_per_row[row_idx])))

        # Start from population mean and reweight it based on whether the row is <=0.5 or >0.5
        reweighted_pop = pop_mean.clone()

        mask_low = (row_probs <= 0.5)
        mask_high = ~mask_low

        # For low entries use y/(1-y); for high entries use (1-y)/y
        reweighted_pop[mask_low] = pop_mean[mask_low] / (1 - pop_mean[mask_low])
        reweighted_pop[mask_high] = (1 - pop_mean[mask_high]) / pop_mean[mask_high]

        # Square-root transformation + cap
        reweighted_pop = torch.sqrt(reweighted_pop)
        reweighted_pop = torch.clamp(reweighted_pop, max=10_000)

        # Allocate step mass proportionally to reweighted_pop
        total_weight = reweighted_pop.sum()
        final_probs = steps * (reweighted_pop / total_weight)

        # Enforce probability mass <= 1 via iterative redistribution
        overflow_mask = (final_probs > 1)
        underflow_mask = (final_probs < 1)
        counter = 0

        while (overflow_mask.any()) and counter < 10:
            leftover = (final_probs[overflow_mask] - 1).sum()
            final_probs[overflow_mask] = 1

            new_sum = reweighted_pop[underflow_mask].sum()
            final_probs[underflow_mask] = (
                leftover * (reweighted_pop[underflow_mask]) / new_sum
                + final_probs[underflow_mask]
            )

            overflow_mask = (final_probs > 1)
            underflow_mask = (final_probs < 1)
            counter += 1

        # Flip the "high" side probabilities
        final_probs[mask_high] = 1.0 - final_probs[mask_high]

        # Clamp away from {0,1}
        final_probs = torch.clamp(final_probs, 1e-5, 1 - 1e-5)

        input_probs[row_idx] = final_probs

    if single_row:
        input_probs = input_probs.squeeze(0)
    return input_probs


def modify_tensor_population(input_probs: torch.Tensor, population_probs: torch.Tensor) -> torch.Tensor:
    """
    Replace entries in input_probs with population_probs wherever the population
    is "more confident" (max(p,1-p) larger).

    Args:
        input_probs: shape (N, D) or (D,)
        population_probs: shape (D,)

    Returns:
        Modified input_probs with original shape preserved.
    """
    single_row = False
    if input_probs.dim() == 1:
        input_probs = input_probs.unsqueeze(0)
        single_row = True

    input_conf = torch.max(input_probs, 1 - input_probs)          # (N, D)
    pop_conf = torch.max(population_probs, 1 - population_probs)  # (D,)

    pop_conf_expanded = pop_conf.unsqueeze(0).expand_as(input_conf)
    replace_mask = pop_conf_expanded > input_conf

    pop_expanded = population_probs.unsqueeze(0).expand_as(input_probs)
    input_probs[replace_mask] = pop_expanded[replace_mask]

    if single_row:
        input_probs = input_probs.squeeze(0)

    return input_probs


def randomly_select_vectors(candidate_vectors: torch.Tensor, index_lists) -> torch.Tensor:
    """
    Given:
      candidate_vectors: tensor of candidate vectors, shape (N, d)
      index_lists:
        - either a single list of indices
        - or a list of lists; one index will be sampled from each inner list

    Returns:
      - if only one list provided: a single vector of shape (d,)
      - else: stacked tensor of shape (M, d)
    """
    selected = []
    if isinstance(index_lists[0], np.int64):
        index_lists = [index_lists]

    for indices in index_lists:
        chosen_index = random.choice(indices)
        selected.append(candidate_vectors[chosen_index])

    if len(selected) == 1:
        return selected[0]
    return torch.stack(selected)


def proportion_mismatch_above_0_5(a, b):
    """
    Proportion of entries where (a > 0.5) XOR (b > 0.5).
    Works for tensors/arrays as long as boolean ops + .float().mean() are valid.
    """
    a_above = (a > 0.5)
    b_above = (b > 0.5)
    mismatches = a_above != b_above
    return mismatches.float().mean()


def count_appearences(train_vectors, fallback_indices, reference_vector):
    """
    Counts how often a particular feature (hard-coded index 2239) differs from
    reference_vector among "nearby" vectors (<=100 mismatches).

    If there are not enough nearby vectors (<=10), fall back to looking up a set
    of indices (fallback_indices) into train_vectors.
    """
    feature_of_interest = 2239

    num_flips = 0
    num_neighbors = 0
    initial_value = reference_vector[feature_of_interest]

    # Primary: scan all train_vectors and find those within <=100 mismatches
    for candidate in train_vectors:
        if torch.sum(candidate != reference_vector) <= 100:
            num_neighbors += 1
            if candidate[feature_of_interest] != reference_vector[feature_of_interest]:
                num_flips += 1

    if num_neighbors > 10:
        flip_rate = num_flips / num_neighbors
    else:
        # Fallback: use the provided indices list
        num_neighbors = 10
        num_flips = 0
        for idx in fallback_indices:
            if train_vectors[idx][feature_of_interest] != reference_vector[feature_of_interest]:
                num_flips += 1
        flip_rate = num_flips / num_neighbors

    return flip_rate, initial_value


class ValDataset(Dataset):
    """
    Dataset used in 'big_test' that returns:
      - transformed input
      - number_changed (flip_rate for feature 2239)
      - initial_state (original value at feature 2239)
    """
    def __init__(self, samples, num_samples, baselines, weights, st_labels, train_ref, indices, pop_mean):
        super(ValDataset, self).__init__()
        self.num_samples = num_samples

        self.samples_tensor = torch.tensor(samples, dtype=torch.float32)
        self.baselines_tensor = torch.tensor(baselines, dtype=torch.float32)
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)

        self.st_labels = st_labels
        self.train_ref_tensor = torch.tensor(train_ref, dtype=torch.float32)
        self.indices = indices
        self.pop_mean_tensor = torch.tensor(pop_mean, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Estimate how often feature 2239 flips among close neighbors
        flip_rate, initial_state = count_appearences(
            self.train_ref_tensor,
            self.indices[idx],
            self.samples_tensor[idx],
        )

        # Apply fixed "shared proportion" modification (self-self)
        noised_sample, shared_prop = fixed_modified(self.samples_tensor[idx], self.samples_tensor[idx])
        shared_prop = torch.tensor(shared_prop, dtype=torch.float32)

        # Apply advanced population-based transformation
        pop_modified = modify_advanced(noised_sample, self.pop_mean_tensor, shared_prop)

        return pop_modified, flip_rate, initial_state


class SampleDataset(Dataset):
    """
    Main training/validation dataset. Returns:
      pop_modified, target_vector, baseline_vector, sample_weight, old_sample
    """
    def __init__(self, samples, num_samples, baselines, weights, st_labels, train_ref, indices, pop_mean):
        super(SampleDataset, self).__init__()
        self.num_samples = num_samples

        self.samples_tensor = torch.tensor(samples, dtype=torch.float32)
        self.baselines_tensor = torch.tensor(baselines, dtype=torch.float32)
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)

        self.st_labels = st_labels
        self.train_ref_tensor = torch.tensor(train_ref, dtype=torch.float32)
        self.indices = indices
        self.pop_mean_tensor = torch.tensor(pop_mean, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Choose a "target/reference" genome vector from train_ref using an index list for this sample
        chosen_reference = randomly_select_vectors(self.train_ref_tensor, self.indices[idx])

        # "Noise" the sample based on how similar it is to the chosen reference
        noised_sample, shared_prop = modify_tensor_based_on_shared_elements(self.samples_tensor[idx], chosen_reference)
        old_sample = noised_sample.clone()

        # Apply advanced population-based transform
        pop_modified = modify_advanced(noised_sample, self.pop_mean_tensor, shared_prop)

        # Return tuple used by training loop
        return pop_modified, chosen_reference, self.baselines_tensor[idx], self.weights_tensor[idx], old_sample


class ResidualBlock(nn.Module):
    def __init__(self, num_features, intermediate_features=200, num_intermediate_layers=0):
        super(ResidualBlock, self).__init__()

        # Project input -> intermediate space, normalize, nonlinearity
        self.fc1 = nn.Linear(num_features, intermediate_features)
        self.bn1 = nn.BatchNorm1d(intermediate_features)

        # Optional stack of intermediate linear layers
        self.intermediate_layers = nn.ModuleList([
            nn.Linear(intermediate_features, intermediate_features)
            for _ in range(num_intermediate_layers)
        ])
        self.intermediate_bns = nn.ModuleList([
            nn.BatchNorm1d(intermediate_features)
            for _ in range(num_intermediate_layers)
        ])

        # Note: fc2 exists in original code but is not used in forward; kept unchanged.
        self.fc2 = nn.Linear(intermediate_features, intermediate_features)

        # Project back to num_features for residual add
        self.fc3 = nn.Linear(intermediate_features, num_features)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        identity = x

        # Original code uses tanh on the input to the block
        out = F.tanh(x)

        # First linear block
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        # Intermediate layers
        for layer, bn in zip(self.intermediate_layers, self.intermediate_bns):
            out = layer(out)
            out = bn(out)
            out = F.leaky_relu(out)

        # Return to original dimension and add residual
        out = self.fc3(out)
        out += identity
        return out


class InteractionBlock(nn.Module):
    def __init__(self, num_features, intermediate_features=200):
        super(InteractionBlock, self).__init__()
        self.fc1 = nn.Linear(num_features, intermediate_features)  # kept, though unused in forward
        self.fc2 = nn.Linear(intermediate_features, num_features)  # kept to preserve exact module fields
        self.ln1 = nn.LayerNorm(num_features)                      # kept, unused in forward
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        identity = x
        out = F.tanh(x)

        # NOTE: original forward applies fc2 directly to out (shape mismatch unless intermediate_features==num_features).
        # We do NOT change behavior; we only keep names and comments.
        out = self.fc2(out)

        out += identity
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_features, config):
        super(CustomResNet, self).__init__()
        self.layers = nn.ModuleList()

        for layer_cfg in config:
            if layer_cfg["type"] == "leaky":
                self.layers.append(
                    ResidualBlock(
                        num_features,
                        layer_cfg.get("intermediate_features", 200),
                        layer_cfg.get("num_intermediate_layers", 1),
                    )
                )
            elif layer_cfg["type"] == "linear":
                self.layers.append(
                    InteractionBlock(
                        num_features,
                        layer_cfg.get("intermediate_features", 200),
                    )
                )
            elif layer_cfg["type"] == "token":
                # TokenizedTransformerBlock is referenced in original code but not defined here.
                # Keeping this branch unchanged.
                block = TokenizedTransformerBlock(
                    num_features,
                    layer_cfg.get("token_size"),
                    layer_cfg.get("intermediate_features"),
                )
                self.layers.append(block)

    def forward(self, x):
        # Work in logit space (with epsilon) then map back through sigmoid at the end
        out = torch.logit(x, eps=1e-5)
        for layer in self.layers:
            out = layer(out)
        out = sigmoid_with_epsilon(out)
        return out


def focal_loss(inputs, targets, gamma):
    """Focal loss wrapper (BCE on probabilities; assumes inputs already in [0,1])."""
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    pt = torch.where(targets == 1, inputs, 1 - inputs)
    modulating_factor = (1 - pt) ** gamma
    return modulating_factor * bce_loss


def perform_validation(model, val_dataloader, baseline, pop_mean, validation_picks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    inv_loss_ratios_vs_guess = []
    inv_loss_ratios_vs_data = []

    correctness_vs_guess = []
    correctness_vs_data = []
    correctness_model = []

    with torch.no_grad():
        for batch_idx, (data, target, baseline, weights, guess) in enumerate(val_dataloader):
            data = data.to(device)
            target = target.to(device)
            baseline = baseline.to(device)
            weights = weights.to(device)
            guess = guess.to(device)

            output = model(data.float())
            loss = focal_loss(output, target.float(), 0)

            # Model correctness
            correct_model = ((output > 0.5) == target.byte()).float()
            correctness_model.append(weighted_mean(correct_model.mean(dim=1), weights).detach().cpu())

            # Two different "priors" used as baselines
            guess_logit = torch.logit(guess.float(), eps=1e-5)
            data_logit = torch.logit(data.float(), eps=1e-5)

            prior_from_guess = sigmoid_with_epsilon(guess_logit)
            prior_from_data = sigmoid_with_epsilon(data_logit)

            base_loss_guess = focal_loss(prior_from_guess, target.float(), 0)
            base_loss_data = focal_loss(prior_from_data, target.float(), 0)

            correct_guess = ((prior_from_guess > 0.5) == target.byte()).float()
            correct_data = ((prior_from_data > 0.5) == target.byte()).float()

            correctness_vs_guess.append(weighted_mean(correct_guess.mean(dim=1), weights).detach().cpu())
            correctness_vs_data.append(weighted_mean(correct_data.mean(dim=1), weights).detach().cpu())

            # Normalize per-sample loss relative to baseline
            sample_baseline_loss_guess = base_loss_guess.sum(dim=1).detach()
            sample_baseline_loss_data = base_loss_data.sum(dim=1).detach()

            batch_ratio_guess = geometric_mean(loss.sum(dim=1).detach(), weights) / geometric_mean(sample_baseline_loss_guess, weights)
            batch_ratio_data = geometric_mean(loss.sum(dim=1).detach(), weights) / geometric_mean(sample_baseline_loss_data, weights)

            inv_loss_ratios_vs_guess.append((1 / batch_ratio_guess.detach()).cpu())
            inv_loss_ratios_vs_data.append((1 / batch_ratio_data.detach()).cpu())

    print("base proportion")
    print(np.median(correctness_vs_guess))
    print("ST proportion")
    print(np.median(correctness_vs_data))
    print("model proportion")
    print(np.median(correctness_model))
    print("start")
    print(np.median(inv_loss_ratios_vs_guess))
    print("advanced")

    return np.median(inv_loss_ratios_vs_data)


def perform_test(model, val_dataloader, baseline, pop_mean, validation_picks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    big_predict = None
    with torch.no_grad():
        for batch_idx, (data, target, baseline, weights, guess) in enumerate(val_dataloader):
            data = data.to(device)
            target = target.to(device)
            baseline = baseline.to(device)
            weights = weights.to(device)
            guess = guess.to(device)

            data_logit = torch.logit(data.float(), eps=1e-5)

            output = model(data.float())
            base_output = sigmoid_with_epsilon(data_logit, epsilon=1e-5)
            pop_output = sigmoid_with_epsilon(torch.logit(torch.tensor(pop_mean).to(device).float(), eps=1e-5), epsilon=1e-5)
            st_output = sigmoid_with_epsilon(torch.logit(baseline.float(), eps=1e-5), epsilon=1e-5)

            model_predict = calculate_outcomes(guess.float().cpu(), output.cpu(), target.cpu(), 2239)
            pop_predict = calculate_outcomes(guess.float().cpu(), target.cpu(), target.cpu(), 2239)
            st_predict = calculate_outcomes(guess.float().cpu(), st_output.cpu(), target.cpu(), 2239)
            base_predict = calculate_outcomes(guess.float().cpu(), base_output.cpu(), target.cpu(), 2239)

            big_predict = np.concatenate((model_predict, pop_predict, st_predict, base_predict), axis=1)
            print(big_predict)

    return big_predict


def big_test(model, test_dataloader, output_csv="output.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["data_2239", "number_of_things", "initial_state"])

        with torch.no_grad():
            for data, number_of_things, initial_state in test_dataloader:
                data = data.to(device).float()
                number_of_things = number_of_things.to(device)
                initial_state = initial_state.to(device)

                outputs = model(data)

                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    value_data_2239 = outputs[i, 2239].item()
                    value_number_of_things = number_of_things[i].item()
                    value_initial_state = initial_state[i].item()

                    writer.writerow([value_data_2239, value_number_of_things, value_initial_state])

    print(f"Finished writing data to {output_csv}")
    torch.save(model.state_dict(), "test_model.pth")
    quit()
    return None


def train_model(model_config, learning_rate, weight_decay, gamma, train_dataset, val_dataset, test_dataset, pop_mean, num_epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders (kept identical params/behavior)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=20)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=20)

    # Model (kept hard-coded feature count)
    model = CustomResNet(num_features=5598, config=model_config)
    model = DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_score = 0
    no_improvement_count = 0

    l1_lambda = 1e-6
    l2_lambda = 1e-5
    performance_watch = []

    for epoch in range(num_epochs):
        model.train()
        print(epoch)

        inv_batch_losses = []
        loss_ratios_l1 = []
        loss_ratios_l2 = []

        for batch_idx, (data, target, baseline, weights, guess) in enumerate(train_dataloader):
            data = data.to(device)
            target = target.to(device)
            baseline = baseline.to(device)
            weights = weights.to(device)
            guess = guess.to(device)

            optimizer.zero_grad()

            output = model(data.float())

            # Baseline computed from *data* (kept from original)
            data_logit = torch.logit(data.float(), eps=1e-5)
            prior_from_data = sigmoid_with_epsilon(data_logit)

            base_loss = focal_loss(prior_from_data, target.float(), gamma)
            loss = focal_loss(output, target.float(), gamma)

            # Normalize per-sample relative to baseline
            sample_baseline_loss = base_loss.sum(dim=1).detach()
            normalized_loss = loss / sample_baseline_loss[:, None]

            batch_loss = normalized_loss.sum() / len(sample_baseline_loss)

            # Regularization terms
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.square().sum() for p in model.parameters())
            total_loss = batch_loss + l2_lambda * l2_norm + l1_norm * l1_lambda

            loss_ratios_l1.append(batch_loss.detach().cpu() / (l1_lambda * l1_norm).detach().cpu())
            loss_ratios_l2.append(batch_loss.detach().cpu() / (l2_lambda * l2_norm).detach().cpu())

            inv_batch_losses.append((1 / batch_loss.detach()).cpu())

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        print(np.median(inv_batch_losses))
        print("loss ratio l1")
        print(np.median(loss_ratios_l1))
        print("loss ratio l2")
        print(np.median(loss_ratios_l2))

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            val_score = perform_validation(model, val_dataloader, baseline.float(), pop_mean, validation_picks)
            performance_watch.append(val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if len(performance_watch) > 20:
                print("difference")
                print(np.mean(performance_watch[-10:]) - np.mean(performance_watch[-20:-10]))
                if np.mean(performance_watch[-10:]) - np.mean(performance_watch[-20:-10]) <= 0 or epoch >= 500:
                    print(f"Stopping early at epoch {epoch}: Validation score did not improve.")
                    break

            print(f"Epoch {epoch}: Validation Score = {val_score}")

    end_output = []
    big_test(model, test_dataloader)
    quit()

    for i in range(10):
        end_output.append(perform_test(model, val_dataloader, baseline.float(), pop_mean, validation_picks))

    return np.mean(end_output, axis=0)


def cross_validate_model(data, base_data, pop_mean, weights, dates, validation_picks, labels, model_config, learning_rate, weight_decay, gamma, num_epochs=500):
    # Ensure the date column is in datetime format
    dates = pd.to_datetime(dates)

    # Split by year (kept identical)
    train_mask = dates.year <= 2019
    val_mask1 = dates.year == 2019
    val_mask2 = dates.year == 2020

    train_indices = np.where(train_mask)[0]
    val_indices1 = np.where(val_mask1)[0]
    val_indices2 = np.where(val_mask2)[0]

    train_samples = data[train_indices]
    val_samples1 = data[val_indices1]
    val_samples2 = data[val_indices2]

    train_baseline = base_data[train_indices]
    val_baseline1 = base_data[val_indices1]
    val_baseline2 = base_data[val_indices2]

    train_weights = weights[train_indices]
    val_weights1 = weights[val_indices1]
    val_weights2 = weights[val_indices2]

    train_labels = labels[train_indices]
    val_labels1 = labels[val_indices1]
    val_labels2 = labels[val_indices2]

    all_train = np.loadtxt("ten_indices.txt").astype(int)
    all_val = np.loadtxt("val_ten_indices.txt").astype(int)

    filtered_train_samples = []
    filtered_train_baseline = []
    filtered_train_weights = []
    filtered_train_labels = []
    filtered_train_index_lists = []

    for i in range(len(all_train)):
        current_indices = all_train[i]
        if current_indices[0] != -1:
            filtered_train_samples.append(train_samples[i])
            filtered_train_baseline.append(train_baseline[i])
            filtered_train_weights.append(train_weights[i])
            filtered_train_labels.append(train_labels[i])
            filtered_train_index_lists.append(current_indices)

    train_dataset = SampleDataset(
        np.array(filtered_train_samples),
        len(filtered_train_samples),
        np.array(filtered_train_baseline),
        np.array(filtered_train_weights),
        filtered_train_labels,
        np.array(train_samples),
        filtered_train_index_lists,
        pop_mean,
    )

    val_dataset = SampleDataset(
        val_samples1,
        len(val_samples1),
        val_baseline1,
        val_weights1,
        val_labels1,
        np.array(val_samples2),
        all_val,
        pop_mean,
    )

    test_dataset = ValDataset(
        val_samples1,
        len(val_samples1),
        val_baseline1,
        val_weights1,
        val_labels1,
        np.array(val_samples2),
        all_val,
        pop_mean,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_score = train_model(model_config, learning_rate, weight_decay, gamma, train_dataset, val_dataset, test_dataset, pop_mean, num_epochs)

    np.savetxt("mecA.csv", val_score, delimiter=",")
    return val_score


def process_file(filename: str, folder_path: str) -> Tuple[str, np.ndarray]:
    """
    Parse a single genome file and return:
      (sequence_type, binary_vector, date_string)

    The filename convention is assumed to encode sequence type and dates in the
    same way as your original code.
    """
    sequence_type = filename.split("__")[-3].split(".")[0]
    date = filename.split("__")[-1].split(".")[0]

    # If collection date is '0000', derive it from upload year-3 (kept logic)
    if date == "0000":
        date = str(int(filename.split("__")[-2].split(".")[0]) - 3)

    with open(os.path.join(folder_path, filename), "r") as file:
        lines = file.readlines()
        binary_vector = np.array([int(line.strip().split("\t")[0]) for line in lines if line.strip()])

    return sequence_type, binary_vector, date


def prepare_dataset_and_averages(folder_path: str, min_members: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all *.txt genome vectors from folder_path, filters by filename-encoded
    date logic, groups by sequence type, and computes:
      - full dataset of binary vectors
      - per-sample baseline frequency vector (ST mean if available else population mean)
      - population mean (from <2020 subset)
      - sample weights (currently all ones)
      - dates, labels, and validation_picks dict
    """
    starting_filenames = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))
    filenames = []

    # Count files per ST and filter filenames by date logic
    sequence_type_counts = {}
    for filename in starting_filenames:
        seq_type = filename.split("__")[-3].split(".")[0]
        collection_date = filename.split("__")[-1].split(".")[0]
        upload_date = filename.split("__")[-2].split(".")[0]

        # Keep if collection date known OR uploaded before 2020
        if collection_date != "0000" or int(upload_date) < 2020:
            sequence_type_counts[seq_type] = sequence_type_counts.get(seq_type, 0) + 1
            filenames.append(filename)

    valid_sequence_types = {st for st, count in sequence_type_counts.items() if count >= min_members}

    num_cpus = cpu_count()
    process_file_partial = partial(process_file, folder_path=folder_path)

    with Pool(num_cpus) as pool:
        results = pool.map(process_file_partial, filenames)

    dataset_vectors = []
    labels = []
    date_list = []

    # Accumulate sums/counts per label (with "other" bucket)
    sequence_type_data = defaultdict(lambda: {"sum": None, "count": 0})
    label_keys = {}
    label_indices = []
    index = 0
    found = 0

    for sequence_type, binary_vector, date in results:
        if sequence_type in valid_sequence_types:
            dataset_vectors.append(binary_vector)
            labels.append(sequence_type)
            date_list.append(date)
            target_type = sequence_type
        else:
            dataset_vectors.append(binary_vector)
            date_list.append(date)
            labels.append("other")
            target_type = "other"

        if sequence_type_data[target_type]["sum"] is None:
            label_keys[target_type] = found
            label_indices.append([])
            found += 1
            sequence_type_data[target_type]["sum"] = np.zeros_like(binary_vector)

        sequence_type_data[target_type]["sum"] += binary_vector
        sequence_type_data[target_type]["count"] += 1
        label_indices[label_keys[target_type]].append(index)
        index += 1

    # Per-label average frequency and counts (weights dict kept but mostly unused later)
    average_frequencies = {}
    weights = {}
    for sequence_type, data in sequence_type_data.items():
        average_frequencies[sequence_type] = data["sum"] / data["count"]
        weights[sequence_type] = data["count"]

    # Build training-only frequencies (<2020) for "validation_frequencies"
    train_freqs = defaultdict(lambda: {"sum": None, "count": 0})
    validation_picks = {}

    for i in range(len(dataset_vectors)):
        st = labels[i]
        binary_vector = dataset_vectors[i]
        current_date = date_list[i]

        if int(current_date) < 2020:
            if train_freqs[st]["sum"] is None:
                train_freqs[st]["sum"] = np.zeros_like(binary_vector)
                validation_picks[st] = []
            train_freqs[st]["sum"] += binary_vector
            validation_picks[st].append(binary_vector)
            train_freqs[st]["count"] += 1

    validation_frequencies = {}
    for sequence_type, data in train_freqs.items():
        validation_frequencies[sequence_type] = data["sum"] / data["count"]
        weights[sequence_type] = data["count"]

    # Population mean over the per-ST validation frequencies
    population_mean = np.array(list(validation_frequencies.values())).mean(axis=0)

    # Baseline frequency per sample: ST mean if present else population mean
    avg_freq_array = np.array([
        validation_frequencies[label] if label in validation_frequencies else population_mean
        for label in labels
    ])

    avg_weights_array = np.array([1 for label in labels])

    true_data_set = np.array(dataset_vectors)
    return true_data_set, avg_freq_array, population_mean, avg_weights_array, date_list, labels, validation_picks


# --------------------
# Main execution
# --------------------
data, base_data, pop_mean, weights, dates, labels, validation_picks = prepare_dataset_and_averages("all_embeded_genomes")
print(len(data))

feature_size = 100  # kept though not used later in this snippet

model_config = [
    {"type": "linear", "intermediate_features": 5598},
    {"type": "leaky", "intermediate_features": 5598, "num_intermediate_layers": 0},
]

cross_validate_model(
    data,
    base_data,
    pop_mean,
    weights,
    dates,
    validation_picks,
    np.array(labels),
    model_config,
    0.00001,
    0,
    0,
)

