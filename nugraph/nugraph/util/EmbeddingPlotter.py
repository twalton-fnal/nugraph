import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import LabelEncoder
import torch
import umap.umap_ as umap

class CombinedEmbeddingPlot:
    """
    A utility class for visualizing embeddings from two datasets using
    dimensionality reduction techniques (Isomap, t-SNE, or UMAP).

    This class accumulates embeddings and their labels from two datasets,
    computes a joint low-dimensional representation, and provides a simple
    interface for plotting or further analysis. It is particularly useful
    for comparing source and target domains in domain adaptation or
    cross-dataset representation learning.

    Parameters
    ----------
    n_neighbors : int, optional (default=10)
        Number of neighbors to use for neighborhood-based methods (Isomap, UMAP).
    n_components : int, optional (default=2)
        Target dimensionality of the embedding space.
    method : str, optional (default='isomap')
        Dimensionality reduction method to use. Options:
        - 'isomap'
        - 'tsne'
        - 'umap'

    Methods
    -------
    update(x1, y1, x2, y2):
        Add new batch of embeddings and labels from two datasets.
    compute():
        Concatenate and return accumulated embeddings and labels for both datasets.
    reset():
        Clear all stored embeddings and labels.
    _get_embedding(X):
        Apply the chosen dimensionality reduction method to the input data.
    subsample(x, y, max_samples=500):
        Get only a subsample (default=500) of embeddings x and associated labels y to reduce plotting time
    plot_combined(dataset1, labels1, dataset2, labels2, epoch=None, class_names=None):
        Use accumulated embeddings from the two datasets to creat a plot. If provided, use epoch number and class names in the plot.

    Notes
    -----
    - Data from the two datasets are accumulated until `compute()` is called.
    - Useful for analyzing how embeddings from different domains overlap
      or separate in the latent space.
    """

    
    def __init__(self, n_neighbors=10, n_components=2, method='isomap'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.dataset1 = []
        self.labels1 = []
        self.dataset2 = []
        self.labels2 = []
        self.method = method.lower()

    def update(self, x1, y1, x2, y2):
        self.dataset1.append(x1.cpu().detach())
        self.labels1.append(y1.cpu().detach())
        self.dataset2.append(x2.cpu().detach())
        self.labels2.append(y2.cpu().detach())

    def compute(self):
        dataset1 = torch.cat(self.dataset1, dim=0)
        labels1 = torch.cat(self.labels1, dim=0)
        dataset2 = torch.cat(self.dataset2, dim=0)
        labels2 = torch.cat(self.labels2, dim=0)
        return dataset1, labels1, dataset2, labels2

    def reset(self):
        self.dataset1 = []
        self.labels1 = []
        self.dataset2 = []
        self.labels2 = []

    def _get_embedding(self, X):
        if self.method == 'isomap':
            return Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components).fit_transform(X)
        elif self.method == 'tsne':
            return TSNE(n_components=self.n_components, perplexity=30, random_state=42).fit_transform(X)
        elif self.method == 'umap':
            return umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=0.1, random_state=42).fit_transform(X)
        else:
            raise ValueError(f"Unsupported method: {self.method}. Choose from 'isomap', 'tsne', or 'umap'.")

    # Randomly sample 1000 embeddings from each domain (or fewer if not enough samples)
    def subsample(self, x, y, max_samples=500):
        num = x.shape[0]
        if num > max_samples:
            idx = torch.randperm(num)[:max_samples]
            return x[idx], y[idx]
        else:
            return x, y

    def plot_combined(self, dataset1, labels1, dataset2, labels2, epoch=None, class_names=None):
        dataset1_np = dataset1.cpu().detach().numpy()
        labels1_np = labels1.cpu().detach().numpy()
        dataset2_np = dataset2.cpu().detach().numpy()
        labels2_np = labels2.cpu().detach().numpy()

        combined_data = np.concatenate([dataset1_np, dataset2_np], axis=0)
        combined_labels = np.concatenate([labels1_np, labels2_np])

        combined_embedding = self._get_embedding(combined_data)

        embedding1 = combined_embedding[:len(dataset1)]
        embedding2 = combined_embedding[len(dataset1):]

        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(combined_labels)))]
        elif len(class_names) <= max(combined_labels):
            raise ValueError("The class_names list must include a name for each class index in the data.")

        unique_classes = np.unique(combined_labels)
        colors = plt.cm.inferno(np.linspace(0, 1, len(unique_classes)))
        class_color_map = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}

        fig, ax = plt.subplots(figsize=(12, 10))

        for cls in np.unique(labels1_np):
            mask = labels1_np == cls
            ax.scatter(
                embedding1[mask, 0], embedding1[mask, 1],
                label=f"Source - {class_names[cls]}", color=class_color_map[cls],
                marker='o', s=100, edgecolor='k', alpha=0.8
            )

        for cls in np.unique(labels2_np):
            mask = labels2_np == cls
            ax.scatter(
                embedding2[mask, 0], embedding2[mask, 1],
                label=f"Target - {class_names[cls]}", color=class_color_map[cls],
                marker='^', s=150, edgecolor='k', alpha=0.8
            )

        ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), framealpha=1.0)
        title = f"{self.method.upper()} Embeddings of Source and Target Datasets"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        ax.set_title(title, fontsize=16)

        plt.close(fig)
        return fig
