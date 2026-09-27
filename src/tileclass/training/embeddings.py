"""Diagnostics for a VICReg backbone's embeddings, used by the Classify
Tiles page's "Explore Embeddings" group: `extract_embeddings` runs the
frozen backbone, `tsne_2d` lays the result out for a scatter plot, and
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


def tsne_2d(embeddings, perplexity=30.0, seed=0):
    """Nonlinear, neighborhood-preserving 2D projection of `embeddings`
    (N, D) via scikit-learn's t-SNE -- unlike a linear projection like PCA,
    a match for what `knn_accuracy` (also a local-neighborhood notion of
    separability) is actually measuring: two well-separated clusters in
    high-dim space can land on top of each other after a linear projection
    if the separating direction isn't one of the top-2 highest-variance
    ones, even though a nonlinear method would show them cleanly apart.

    `perplexity` is clamped below `len(embeddings)` (scikit-learn requires
    it) and roughly to `len(embeddings) // 4`, so a small annotated pool
    doesn't force an oversized perplexity relative to its own size --
    t-SNE gets unreliable/collapses points together long before hitting
    sklearn's hard error. `seed` fixes the (otherwise stochastic) layout
    so repeated calls on the same embeddings are reproducible, though the
    layout still isn't comparable run-to-run -- distances between clusters,
    and cluster sizes, aren't meaningful in a t-SNE plot, only which points
    cluster together."""
    from sklearn.manifold import TSNE

    n = len(embeddings)
    effective_perplexity = max(1.0, min(perplexity, (n - 1) / 3, n // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        init="pca",
    )
    return tsne.fit_transform(embeddings)


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
