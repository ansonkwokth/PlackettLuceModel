# Plackett-Luce Model

A Python package for implementing and using the Plackett-Luce model for ranking.

## Features
- Fit the Plackett-Luce model to rankings data
- Predict ranking probabilities
- Generate synthetic datasets

## Example **`examples.ipynb`**
Example script showing how to use the package.

```python
from plackett_luce import datasets as ds
from plackett_luce.model import PlackettLuceModel
from plackett_luce.utils import EarlyStopper

# Generate synthetic data
X_train, rankings_train = ds.generate_data(1000, 15)

# Initialize and fit the model
model = PlackettLuceModel(score_model=custom_nn)
model.fit(X_train, rankings_train)

