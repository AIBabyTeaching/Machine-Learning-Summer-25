"""Plotting helpers for visualizing machine learning results.

Each function includes a brief demonstration in the ``__main__`` block
and extensive documentation. ``matplotlib`` and ``seaborn`` are used for
plotting. References:

- matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
- seaborn: https://seaborn.pydata.org/api.html
- confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion(y_true, y_pred, labels=None) -> None:
    """Display a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values from the model.
    labels : list, optional
        Class label ordering passed to
        :func:`sklearn.metrics.confusion_matrix`.

    Returns
    -------
    None
        The function creates a heatmap plot for visual inspection.
    """
    # Compute confusion matrix using scikit-learn utility
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Use a blue color map for better contrast
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")  # x-axis label
    plt.ylabel("Actual")     # y-axis label
    plt.tight_layout()
    plt.show()


def plot_regression_results(y_true, y_pred) -> None:
    """Scatter plot comparing true and predicted values.

    Parameters
    ----------
    y_true : array-like
        Observed target values.
    y_pred : array-like
        Predicted values from a regression model.

    Returns
    -------
    None
        Displays a scatter plot with a reference ``y=x`` line.
    """
    # Scatter actual vs predicted values
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Compute global min/max for the reference diagonal
    min_val = np.min([y_true, y_pred])
    max_val = np.max([y_true, y_pred])

    # Dashed red line representing perfect predictions
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_clusters(X: np.ndarray, labels: np.ndarray) -> None:
    """Scatter plot for 2-D clustered data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        Two-dimensional feature matrix.
    labels : np.ndarray, shape (n_samples,)
        Cluster labels for each sample.

    Returns
    -------
    None
        Displays a scatter plot with color-coded clusters.
    """
    if X.shape[1] != 2:
        raise ValueError("X must have exactly 2 columns for plotting")

    # Categorical color palette for distinct cluster colors
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="tab10")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demonstrate basic usage of the helpers
    rng = np.random.RandomState(0)

    # Demo confusion matrix
    true = rng.randint(0, 2, size=20)
    preds = rng.randint(0, 2, size=20)
    print("Plotting confusion matrix...")
    plot_confusion(true, preds)

    # Demo regression results
    y = rng.randn(50)
    y_hat = y + rng.normal(scale=0.5, size=len(y))
    print("Plotting regression scatter...")
    plot_regression_results(y, y_hat)

    # Demo clustering
    X = rng.randn(30, 2)
    labels = rng.randint(0, 3, size=30)
    print("Plotting clusters...")
    plot_clusters(X, labels)

