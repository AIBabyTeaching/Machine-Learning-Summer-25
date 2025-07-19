import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, labels=None):
    """Plot a confusion matrix using seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_regression_results(y_true, y_pred):
    """Scatter true vs predicted values with a y=x reference line."""
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()


def plot_clusters(X, labels):
    """Plot 2-D cluster assignments."""
    if X.shape[1] > 2:
        raise ValueError("X must have exactly 2 columns for plotting")
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='tab10')
    plt.show()
