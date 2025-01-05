import torch
import torch.nn as nn
import torch.optim as optim
from plackett_luce.utils import EarlyStopper



class PlackettLuceModel(nn.Module):
    """
    Implementation of the Plackett-Luce model for ranking using PyTorch
    and gradient descent for Maximum Likelihood Estimation (MLE).
    """

    def __init__(self, score_model=None, early_stopper=None):
        """
        Initialize the Plackett-Luce model.

        Args:
            input_dim (int): Dimension of input features.
            score_model (nn.Module): Neural network model for scoring items.
                Must have a 1D output.
            early_stopper (EarlyStopper): Instance of EarlyStopper for training.
                If None, a default EarlyStopper is used.
        """
        super(PlackettLuceModel, self).__init__()

        self.score_model = score_model
        # Check the output dimension of the model's last layer
        if isinstance(self.score_model, nn.Module):
            # Check if the last layer of the model has a 1D output (out_features = 1)
            if hasattr(self.score_model, "[-1]") and isinstance(self.score_model[-1], nn.Linear):
                last_layer = self.score_model[-1]
                if last_layer.out_features != 1:
                    raise ValueError("Score model's last layer must have out_features=1 (1D output).")

        # Early stopper
        self.early_stopper = early_stopper if early_stopper is not None else EarlyStopper(patience=10, min_delta=0.01)




    def forward(self, X, item_mask=None):
        """
        Forward pass to compute scores for items.

        Args:
            X (torch.Tensor): Feature matrix for items (batch_size, num_items, input_dim).
            item_mask (torch.Tensor): Binary mask for valid items (batch_size, num_items).

        Returns:
            torch.Tensor: Scores for each item (batch_size, num_items).
        """
        batch_size, num_items, _ = X.shape
        X_flat = X.view(batch_size * num_items, -1)  # Flatten items in the batch
        scores = self.score_model(X_flat).view(batch_size, num_items)  # Reshape back

        if item_mask is not None:
            scores[item_mask == 0] = -torch.inf
            # scores = scores * item_mask  # Apply mask

        return scores



    def _compute_log_likelihood(self, scores, rankings, item_mask=None, top_k=None):
        """
        Compute the negative log-likelihood for the Plackett-Luce model.

        Args:
            scores (torch.Tensor): Scores for items (batch_size, num_items).
            rankings (torch.Tensor): Observed rankings (batch_size, num_items).
            item_mask (torch.Tensor, optional): A mask tensor indicating which items to consider in the likelihood.
            top_k (int, optional): The number of top items to consider in the likelihood. If None, all items are used.

        Returns:
            torch.Tensor: Negative log-likelihood (scalar).
        """
        batch_size, num_items = scores.shape
        nll = 0.0
        for b in range(batch_size):            
            # Apply item mask if provided
            # if item_mask is not None:
            ranking = rankings[b] * item_mask[b]  # Ranking for sample b
            ranking_scores = scores[b, ranking]  # Scores in the order of the ranking

            for i in range(ranking_scores.shape[0]):
                if top_k is not None and i == top_k: break
                denominator = torch.logsumexp(ranking_scores[i:], dim=0)
                nll -= ranking_scores[i] - denominator
                
        return nll / batch_size



    def fit(self, X, rankings, lr=0.01, epochs=100, early_stopper=None, top_k=None, item_mask=None, verbose=True):
        """
        Fit the Plackett-Luce model to the given rankings using gradient descent.

        Args:
            X (torch.Tensor): Feature matrix for items (batch_size, num_items, input_dim).
            rankings (torch.Tensor): Observed rankings (batch_size, num_items).
            lr (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
            early_stopper (EarlyStopper, optional): Early stopping criterion.
            top_k (int, optional): Number of top items to consider in the likelihood.
            item_mask (torch.Tensor, optional): Mask for considering specific items in the likelihood.
            verbose (bool): Whether to print training progress.

        Returns:
            None
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            scores = self.forward(X)  # Compute scores for the batch
            nll = self._compute_log_likelihood(scores, rankings, item_mask=item_mask, top_k=top_k)  # Compute NLL with the new parameters
            nll.backward()  # Backpropagate the gradients
            optimizer.step()  # Update parameters

            # Early stopping check
            if self.early_stopper is not None and self.early_stopper.early_stop(nll.item()):
                print(f"Early stopping at epoch {epoch + 1} with NLL {nll.item():.4f}")
                break

            # Verbose output
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Negative Log-Likelihood: {nll.item():.4f}")



    def predict(self, X, item_mask=None, top_k=None):
        """
        Predict rankings for a batch of samples based on fitted scores.

        Args:
            X (torch.Tensor): Feature matrix for items (batch_size, max_num_items, input_dim).
            item_mask (torch.Tensor): Mask indicating valid items (batch_size, max_num_items), with 1 for valid and 0 for padding.
            top_k (int): Only return the top-k rankings (default: all).

        Returns:
            list of lists: Predicted rankings for each sample in the batch.
        """
        with torch.no_grad():
            scores = self.forward(X, item_mask=item_mask)  # Compute scores
            rankings = torch.argsort(scores, dim=1, descending=True)  # Sort by scores
            if top_k is not None:
                rankings = rankings[:, :top_k]
            return rankings.tolist()



    def predict_given_config(self, X, config, item_mask=None):
        """
        Predict the probability of a given ranking configuration.

        Args:
            X (torch.Tensor): Feature matrix for items (batch_size, max_num_items, input_dim).
            config (list): Specific ranking configuration (e.g., [1, 3, 4]).
            item_mask (torch.Tensor): Mask indicating valid items (batch_size, max_num_items), with 1 for valid and 0 for padding.

        Returns:
            list: Probabilities of the given ranking configuration for each sample in the batch.
        """
        with torch.no_grad():
            scores = self.forward(X, item_mask=item_mask)  # Compute scores
            probs = []
            for b in range(scores.shape[0]):
                config_scores = scores[b, config]
                prob = 1.0
                for i in range(len(config)):
                    denominator = torch.logsumexp(config_scores[i:], dim=0).exp()
                    prob *= (config_scores[i].exp() / denominator).item()
                probs.append(prob)
            return probs


