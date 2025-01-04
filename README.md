# Plackett-Luce Model Implementation

## Overview
This repository provides an implementation of the Plackett-Luce model for ranking using PyTorch. The model supports:

- Flexible neural network (NN) structures for scoring.
- Training with negative log-likelihood (NLL) loss.
- Early stopping to prevent overfitting.
- Customizable parameters for model training.
## Mathematical Definition

The probability of a ranking $\pi = (\pi_1, \pi_2, \ldots, \pi_n)$ is given by:

$$P(\pi | s_1, s_2, \ldots, s_n) = \prod_{k=1}^n \frac{\exp(s_{\pi_k})}{\sum_{j=k}^n \exp(s_{\pi_j})}$$

Where:
- $s_i$ is the score assigned to item $i$.
- $\pi_k$ is the $k$-th ranked item in the ranking $\pi$.
- $\sum_{j=k}^n \exp(s_{\pi_j})$ represents the sum of exponentiated scores for items ranked $k$ to $n$.

The model ensures that higher scores result in higher ranks, as items with higher scores contribute more to the product of probabilities.


## Features
- **Flexible NN Input**: Pass your custom neural network for scoring, as long as it outputs 1D scores.
- **Early Stopping**: Integrated early stopping mechanism to halt training based on validation performance.
- **Top-K Training**: Train the model while focusing on the top-K items in rankings.

## Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/ansonkwokth/PlackettLuceModel.git

# Navigate to the project directory
cd PlackettLuceModel

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Data Generation
Use the provided script to generate synthetic data for training and testing:

```python
from plackett_luce import datasets as ds

# Generate training and testing data
X_train, rankings_train = ds.generate_data(1000, 10)
X_test, rankings_test = ds.generate_data(500, 10)
```

### Model Definition
Define a custom neural network for scoring:

```python
from torch import nn

class CustomNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 1D output for scoring
        )

    def forward(self, x):
        return self.network(x)
```

### Training the Model

```python
from plackett_luce.model import PlackettLuceModel
from plackett_luce.utils import EarlyStopper

# Initialize the custom NN model and early stopper
nn_model = CustomNNModel(input_dim=15)
custom_early_stopper = EarlyStopper(patience=20, min_delta=0.01)

model = PlackettLuceModel(score_model=nn_model, early_stopper=custom_early_stopper)

model.fit(X_train, rankings_train, lr=0.01, epochs=500, top_k=3)
```

### Prediction

```python
# Predict rankings for the test set
predicted_rankings = model.predict(X_test)
```

## Testing
Unit tests are provided in the `tests` folder. Run the tests using:

```bash
python -m unittest discover -s tests
```

Example:

```bash
python -m unittest tests/test_utils.py
```

## Project Structure

```
plackett_luce/
├── .github/               # GitHub-specific configurations
│   └── workflows/
│       └── ci.yml         # CI pipeline for tests, linting, etc.
├── docs/                  # Documentation files
│   └── index.md           # Documentation entry point
├── plackett_luce/         # Source code of the package
│   ├── __init__.py        # Makes this a package, exposes main API
│   ├── datasets.py        # Functions for loading example datasets
│   ├── model.py           # Core implementation of Plackett-Luce model
│   ├── utils.py           # Utility functions (e.g., data processing)
│   └── tests/             # Unit and integration tests
│       ├── __init__.py    # Test module init
│       ├── test_model.py  # Tests for Plackett-Luce implementation
│       └── test_utils.py  # Tests for utilities
├── .gitignore             # Git ignore file
├── README.md              # Overview of your package
├── examples.ipynb         # Example notebook
├── pyproject.toml         # Modern Python project configuration
├── requirements.txt       # Dependencies (if not using pyproject.toml)
└── setup.py               # Optional, traditional entry for packaging

```
