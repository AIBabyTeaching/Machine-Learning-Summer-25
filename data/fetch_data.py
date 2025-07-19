"""Utility functions to fetch example datasets for the course.

This module provides helper functions to load the bundled CSV datasets
and return them as pandas DataFrames. The data shapes and sources are
listed below:

- `load_iris()` -> (150, 5). The classic Iris dataset from
  https://archive.ics.uci.edu/ml/datasets/iris used for
  classification examples.
- `load_house_prices()` -> (1000, 9). Randomly generated housing
  features for regression tasks.
- `load_moons()` -> (200, 3). Two interleaving moons synthetic
  dataset commonly used for binary classification demonstrations.

These datasets are small and meant for teaching purposes only.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def load_iris() -> pd.DataFrame:
    """Load the Iris dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing four feature columns and a numeric
        ``class`` column. See :class:`pandas.DataFrame` docs for
        usage details.
    """
    path = DATA_DIR / "iris.csv"
    return pd.read_csv(path)


def load_house_prices() -> pd.DataFrame:
    """Load the synthetic housing prices dataset."""
    path = DATA_DIR / "house_prices.csv"
    return pd.read_csv(path)


def load_moons() -> pd.DataFrame:
    """Load the two moons dataset."""
    path = DATA_DIR / "moons.csv"
    return pd.read_csv(path)


if __name__ == "__main__":
    # Demonstrate simple usage when running this module directly
    iris = load_iris()
    print("Iris shape:", iris.shape)
    print(iris.head())
    house = load_house_prices()
    print("House prices shape:", house.shape)
    moons = load_moons()
    print("Moons shape:", moons.shape)

