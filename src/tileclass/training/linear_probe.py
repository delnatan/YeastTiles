"""Linear-probe evaluation of a VICReg backbone's embeddings.

This is the diagnostic design.md's split between embedding quality and
classifier quality is meant to enable: freeze the backbone entirely (no
gradient flows into it at all) and see how far a single linear layer on
top of its frozen output gets. If a linear layer -- the simplest
possible decision boundary -- already separates the annotated
categories well, the embedding space itself carries that information;
if it doesn't, no amount of classifier-head cleverness on top will fix
it, and the problem is upstream in `training.vicreg`.

`knn_accuracy` is a second, gradient-free view of the same question: it
never fits any parameters, so it can't overfit the way even a small
linear probe could on a rare category with only a handful of examples.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import MaskedMicroscopyDataset, class_weights, stratified_split


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


def pca_2d(embeddings):
    """Project `embeddings` (N, D) onto their top-2 principal components
    via SVD -- a dependency-free stand-in for sklearn's PCA, just for a
    quick visual read on class separability."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def tsne_2d(embeddings, perplexity=30.0, seed=0):
    """Nonlinear, neighborhood-preserving 2D projection of `embeddings`
    (N, D) via scikit-learn's t-SNE -- unlike `pca_2d`'s linear/global
    projection, this is a much better match for what `knn_accuracy` (also
    a local-neighborhood notion of separability) is actually measuring:
    two well-separated clusters in high-dim space can land on top of each
    other after a linear projection if the separating direction isn't one
    of the top-2 highest-variance ones, even though a nonlinear method
    would show them cleanly apart.

    `perplexity` is clamped below `len(embeddings)` (scikit-learn requires
    it) and roughly to `len(embeddings) // 4`, so a small annotated pool
    doesn't force an oversized perplexity relative to its own size --
    t-SNE gets unreliable/collapses points together long before hitting
    sklearn's hard error. `seed` fixes the (otherwise stochastic) layout
    so repeated calls on the same embeddings are reproducible, though the
    layout still isn't comparable run-to-run the way `pca_2d`'s
    deterministic axes are -- distances between clusters, and cluster
    sizes, aren't meaningful in a t-SNE plot, only which points cluster
    together."""
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
    no trained parameters at all -- a floor the linear probe should at
    least match."""
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


@dataclass
class LinearProbeParams:
    val_frac: float = 0.2
    seed: int = 0
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-4


@dataclass
class LinearProbeResult:
    val_accuracy: float
    per_class_accuracy: dict = field(default_factory=dict)
    train_count: int = 0
    val_count: int = 0


def train_linear_probe(
    embeddings,
    labels,
    categories,
    params: LinearProbeParams = LinearProbeParams(),
    progress_callback=None,
) -> LinearProbeResult:
    """`embeddings`: (N, D) array from `extract_embeddings`. `labels`:
    length-N list of category names, restricted to `categories`'s
    vocabulary (a linear layer's output width is fixed to it). Trains a
    single `nn.Linear` on frozen embeddings -- no backbone in the loop at
    all, unlike `supervised.train_classifier`'s probe stage, which still
    carries the full EfficientNet forward pass."""
    label_to_idx = {name: i for i, name in enumerate(categories)}
    unknown = sorted(set(labels) - set(label_to_idx))
    if unknown:
        raise ValueError(f"Unrecognized categories: {unknown}")
    int_labels = [label_to_idx[label] for label in labels]

    train_idx, val_idx = stratified_split(
        int_labels, val_frac=params.val_frac, seed=params.seed
    )
    if not train_idx or not val_idx:
        raise ValueError(
            "Not enough embeddings to form a train/validation split -- "
            "annotate more tiles (ideally at least a couple per category) first."
        )

    X = torch.from_numpy(embeddings).float()
    y = torch.tensor(int_labels, dtype=torch.long)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    num_classes = len(categories)
    head = nn.Linear(embeddings.shape[1], num_classes)
    weights = class_weights([int_labels[i] for i in train_idx], num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    head.train()
    for epoch in range(params.epochs):
        perm = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(perm), params.batch_size):
            batch_idx = perm[start : start + params.batch_size]
            optimizer.zero_grad()
            loss = criterion(head(X_train[batch_idx]), y_train[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if progress_callback is not None:
            progress_callback(epoch + 1, params.epochs, epoch_loss / max(1, num_batches))

    head.eval()
    with torch.no_grad():
        val_logits = head(X_val)
    val_preds = val_logits.argmax(dim=1)
    val_accuracy = (val_preds == y_val).float().mean().item()

    per_class_accuracy = {}
    for class_idx, name in enumerate(categories):
        mask = y_val == class_idx
        if mask.sum() == 0:
            continue
        per_class_accuracy[name] = (val_preds[mask] == class_idx).float().mean().item()

    return LinearProbeResult(
        val_accuracy=val_accuracy,
        per_class_accuracy=per_class_accuracy,
        train_count=len(train_idx),
        val_count=len(val_idx),
    )
