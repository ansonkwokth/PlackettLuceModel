import torch
import numpy as np
import random

torch.manual_seed(0)

def secret_scoring_function(features):
    """
    Complex, nonlinear secret scoring function for 15 features.
    Args:
        features (torch.Tensor): Feature matrix of shape (num_items, num_features).

    Returns:
        torch.Tensor: Scores for each item.
    """
    # Nonlinear interaction terms (pairwise interactions across feature groups)
    interaction_term_1 = torch.sum(features[:, :5] * features[:, 5:10], dim=1)  # Interaction between first 10 features
    interaction_term_2 = torch.sum(features[:, 10:] * features[:, :5], dim=1)  # Interaction between last 5 and first 5 features

    # Polynomial terms (sum of squared and cubic features)
    poly_term_2 = torch.sum(features[:, :10] ** 2, dim=1)  # Squared terms for first 10 features
    poly_term_3 = torch.sum(features[:, 10:] ** 3, dim=1)  # Cubic terms for last 5 features

    # Nonlinear activations (ReLU and Sigmoid)
    relu_term = torch.relu(features @ torch.tensor([0.2, -0.3, 0.5, 0.7, -0.1, 0.1, -0.4, 0.6, 0.3, -0.2,
                                                     0.4, 0.8, -0.6, 0.2, -0.5]))  # Weighted sum with ReLU
    sigmoid_term = torch.sigmoid(features @ torch.tensor([-0.3, 0.1, 0.6, -0.4, 0.7, -0.5, 0.2, 0.8, -0.7, 0.1,
                                                          -0.6, 0.3, 0.4, -0.2, 0.5]))  # Weighted sum with Sigmoid

    # Combine all terms and add noise
    scores = interaction_term_1 + interaction_term_2 + 0.4 * poly_term_2 + 0.2 * poly_term_3 + 0.7 * relu_term + 0.5 * sigmoid_term
    scores += 0.1 * torch.randn(features.shape[0])  # Add noise

    return scores



def generate_data(num_samples, num_items):
    """
    Generate data samples for training or testing.
    Args:
        num_samples (int): Number of samples to generate.
        num_items (int): Number of items per sample.
        num_features (int): Number of features per item.

    Returns:
        X_data (torch.Tensor): Feature matrix (num_samples, num_items, num_features).
        rankings (torch.Tensor): True rankings (num_samples, num_items).
    """
    num_features = 15
    X_data = torch.randn(num_samples, num_items, num_features)  # Random features
    rankings = []

    for i in range(num_samples):
        scores = secret_scoring_function(X_data[i])  # Compute secret scores
        ranking = torch.argsort(scores, descending=True)  # Rank items by score (highest first)
        rankings.append(ranking)

    rankings = torch.stack(rankings)  # Stack rankings into a single tensor
    return X_data, rankings





def generate_data_varaible_items(num_samples, num_items_range):
    """
    Generate random data with varying num_items for the given number of samples.

    Parameters:
        ds: Dataset object with a `generate_data` method.
        num_samples (int): Total number of samples to generate.
        num_items_range (tuple): Range (min, max) for `num_items` (inclusive).

    Returns:
        tuple: X_train (list), rankings_train (list), n_features (int)
    """
    # Initialize empty lists to store generated data
    X = []
    rankings = []
    n_features = None  # To be set after the first generation

    # Generate data
    for _ in range(num_samples):
        num_items = random.randint(*num_items_range)  # Randomly pick num_items
        X_temp, rankings_temp = generate_data(num_samples=1, num_items=num_items)
        
        X.extend(X_temp.tolist())
        rankings.extend(rankings_temp.tolist())

    return X, rankings




