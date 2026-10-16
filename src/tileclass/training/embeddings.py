"""Diagnostics for a VICReg backbone's embeddings, used by the Classify
Tiles page's "Explore Embeddings" group: `extract_embeddings` runs the
frozen backbone, `umap_2d` lays the result out for a scatter plot, and
`knn_accuracy` scores how well the annotated categories separate --
gradient-free, so it can't overfit a rare category with only a handful of
examples. If they don't separate, no amount of classifier-head cleverness
on top will fix it, and the problem is upstream in `training.vicreg`."""

from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MaskedMicroscopyDataset, stratified_split


def extract_embeddings(paths, backbone, device, batch_size=64):
    """Deterministic (no augmentation) forward pass through a frozen
    backbone -- one embedding row per path, in the same order as
    `paths`."""
    backbone.eval()
    dataset = MaskedMicroscopyDataset(paths, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(backbone(batch.to(device)).cpu())
    return torch.cat(embeddings, dim=0).numpy()


UMAP_METRICS = ("euclidean", "cosine", "correlation", "manhattan")


def umap_2d(embeddings, n_neighbors=15, min_dist=0.1, metric="euclidean", seed=0):
    """Nonlinear, neighborhood-preserving 2D projection of `embeddings`
    (N, D) via UMAP -- unlike a linear projection like PCA, a match for
    what `knn_accuracy` (also a local-neighborhood notion of separability)
    is actually measuring: two well-separated clusters in high-dim space
    can land on top of each other after a linear projection if the
    separating direction isn't one of the top-2 highest-variance ones.
    Much faster than t-SNE on a large pool, and keeps more of the global
    layout (which clusters sit near which).

    `n_neighbors` trades local detail (small) for global structure
    (large); `min_dist` is how tightly points may pack within a cluster.
    `n_neighbors` is clamped below `len(embeddings)`, so a small annotated
    pool doesn't need a hand-picked value to run at all. `seed` fixes the
    (otherwise stochastic) layout so repeated calls on the same embeddings
    are reproducible -- at the cost of UMAP running single-threaded --
    or None for a faster, parallel, non-reproducible run. Cluster sizes
    and gaps between far-apart clusters still aren't quantitatively
    meaningful, only which points group together."""
    import warnings

    import umap

    n = len(embeddings)
    if n < 3:
        raise ValueError("UMAP needs at least 3 points to lay out.")
    with warnings.catch_warnings():
        # "n_jobs value 1 overridden to 1 by setting random_state" -- the
        # documented cost of a fixed seed, not something to act on.
        warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(int(n_neighbors), n - 1)),
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            # Spectral init needs more points than a tiny pool has.
            init="spectral" if n > 10 else "random",
        )
        return reducer.fit_transform(np.asarray(embeddings, dtype=np.float32))


def knn_accuracy(embeddings, labels, k=5, val_frac=0.2, seed=0):
    """Held-out k-nearest-neighbor vote accuracy in embedding space, with
    no trained parameters at all."""
    train_idx, val_idx = stratified_split(labels, val_frac=val_frac, seed=seed)
    if not train_idx or not val_idx:
        raise ValueError("Not enough embeddings to form a train/validation split.")

    labels = np.asarray(labels)
    train_emb = torch.from_numpy(embeddings[train_idx]).float()
    val_emb = torch.from_numpy(embeddings[val_idx]).float()
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    k = min(k, train_emb.shape[0])
    dists = torch.cdist(val_emb, train_emb)
    nearest = dists.topk(k, largest=False).indices.numpy()

    correct = 0
    for neighbor_idxs, true_label in zip(nearest, val_labels):
        votes = Counter(train_labels[i] for i in neighbor_idxs)
        predicted = votes.most_common(1)[0][0]
        correct += predicted == true_label
    return correct / len(val_labels)
