import torch
import numpy as np

class EarlyStopper:
    """
    Early stopping utility to monitor validation loss and halt training when
    improvements fall below a minimum threshold for a defined number of epochs.
    """
    def __init__(self, patience=10, min_delta=0.01):
        """
        Args:
            patience (int): Number of epochs to wait for significant improvement.
            min_delta (float): Minimum improvement threshold (percentage of current best loss).
        """
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")
        if min_delta <= 0:
            raise ValueError("min_delta must be a positive number.")

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, current_loss):
        """
        Check whether to stop early based on the validation loss.

        Args:
            current_loss (float): Current validation loss.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        # Check if the current loss has improved significantly
        if current_loss < self.best_loss * (1 - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0  # Reset counter as improvement is found
        else:
            self.counter += 1  # No significant improvement
            if self.counter >= self.patience:
                return True  # Trigger early stopping if no improvement for 'patience' epochs

        return False






class DataLoader:
    def __init__(self):
        pass

    def transform(self, df):
        """
        Transform the input dataframe into feature matrices and rankings, 
        ensuring proper padding for inconsistent item counts across samples.
        
        Args:
            df (pandas.DataFrame): Input dataframe with columns 'ID', 'rank', and other features.
        
        Returns:
            tuple: A tuple containing:
                - X (torch.Tensor): Feature matrix of shape (n_samples, max_items, n_features)
                - rankings (torch.Tensor): Ranking array of shape (n_samples, max_items)
                - mask (torch.Tensor): Mask indicating valid rankings (n_samples, max_items)
        """
        X, rankings = self._transform_to_lists(df)
        min_items, max_items = self._get_min_max_item_lengths(rankings)
        return self._get_masked(X, rankings, max_items)

    def _transform_to_lists(self, df):
        """
        Transforms the dataframe into feature lists and rankings.
        
        Args:
            df (pandas.DataFrame): Input dataframe with 'ID', 'rank', and other features.
        
        Returns:
            tuple: X (list) and rankings (list) of transformed feature values and rankings.
        """
        X, rankings = [], []
        for id in df.ID.unique():
            dfi = df[df.ID == id].drop("ID", axis=1)
            rank = dfi.pop("rank").to_list()
            features = dfi.values.tolist()
            X.append(features)
            rankings.append(rank)
        return X, rankings

    def _get_min_max_item_lengths(self, rankings):
        """
        Calculate the minimum and maximum number of items across all rankings.
        
        Args:
            rankings (list): List of rankings for each sample.
        
        Returns:
            tuple: min_items and max_items representing the length of the shortest 
                   and longest ranking lists.
        """
        min_items = min(len(rank) for rank in rankings)
        max_items = max(len(rank) for rank in rankings)
        return min_items, max_items

    def _get_masked(self, X, rankings, max_items):
        """
        Pads the feature and ranking arrays to the same length, and creates a mask 
        indicating where the rankings are valid (non-NaN).
        
        Args:
            X (list): Feature matrices.
            rankings (list): Rankings.
            max_items (int): Maximum number of items to pad to.
        
        Returns:
            tuple: A tuple containing:
                - X (torch.Tensor): Padded feature matrix.
                - rankings (torch.Tensor): Padded rankings.
                - mask (torch.Tensor): Mask indicating valid entries.
        """
        n_samples = len(rankings)
        n_features = len(X[0][0]) if X and X[0] else 0  # Handle empty input gracefully

        # Pad the rankings and feature matrices to max_items length
        for i in range(n_samples):
            n_empty = max_items - len(X[i])
            if n_empty > 0:
                rankings[i].extend([np.nan] * n_empty)
                X[i].extend([[np.nan] * n_features] * n_empty)

        # Convert lists to numpy arrays
        X = np.array(X, dtype=float)
        rankings = np.array(rankings, dtype=float)

        # Handle NaNs by creating a mask and replacing NaN values with 0 in X
        mask = ~np.isnan(rankings)
        X[np.isnan(X)] = 0

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        rankings = torch.tensor(rankings, dtype=torch.float32)  # Use float32 for rankings to handle NaNs
        mask = torch.tensor(mask, dtype=torch.int32)

        return X, rankings, mask
