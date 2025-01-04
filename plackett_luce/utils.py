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

