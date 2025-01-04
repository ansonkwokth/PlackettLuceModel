import unittest
from plackett_luce.utils import EarlyStopper  # Updated import path

class TestEarlyStopper(unittest.TestCase):
    
    def test_patience_zero(self):
        """Test that passing patience = 0 raises a ValueError."""
        with self.assertRaises(ValueError):
            EarlyStopper(patience=0, min_delta=0.01)
    
    def test_patience_negative(self):
        """Test that passing a negative patience raises a ValueError."""
        with self.assertRaises(ValueError):
            EarlyStopper(patience=-5, min_delta=0.01)
    
    def test_min_delta_zero(self):
        """Test that passing min_delta = 0 raises a ValueError."""
        with self.assertRaises(ValueError):
            EarlyStopper(patience=10, min_delta=0)
    
    def test_min_delta_negative(self):
        """Test that passing a negative min_delta raises a ValueError."""
        with self.assertRaises(ValueError):
            EarlyStopper(patience=10, min_delta=-0.01)

    def test_early_stop(self):
        """Test the early stopping logic when improvement is below the threshold."""
        early_stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        # Simulate a scenario where the validation loss does not improve
        self.assertFalse(early_stopper.early_stop(1.0))  # No improvement
        self.assertFalse(early_stopper.early_stop(1.0))  # Still no improvement
        self.assertFalse(early_stopper.early_stop(1.0))  # Still no improvement
        self.assertTrue(early_stopper.early_stop(1.0))  # Early stop after 3 epochs with no improvement

    def test_improvement(self):
        """Test that early stopping is not triggered when there is improvement."""
        early_stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        # Simulate a scenario where the validation loss improves
        self.assertFalse(early_stopper.early_stop(1.0))  # Initial loss
        self.assertFalse(early_stopper.early_stop(0.99))  # Slight improvement
        self.assertFalse(early_stopper.early_stop(0.98))  # Continued improvement
        self.assertFalse(early_stopper.early_stop(0.97))  # Still improving (no early stop)

    def test_no_improvement_for_patience(self):
        """Test that early stopping triggers after no improvement for 'patience' epochs."""
        early_stopper = EarlyStopper(patience=3, min_delta=0.01)
        
        # Simulate the case where there's no improvement for 3 epochs
        self.assertFalse(early_stopper.early_stop(1.0))  # No improvement
        self.assertFalse(early_stopper.early_stop(1.0))  # Still no improvement
        self.assertFalse(early_stopper.early_stop(1.0))  # Still no improvement
        self.assertTrue(early_stopper.early_stop(1.0))  # No improvement after 3 epochs, trigger early stop

if __name__ == "__main__":
    unittest.main()

