"""Interface every auto-annotator plugs into.

Kept minimal on purpose: a classifier just maps a list of crop-image
paths to predicted (label, confidence) pairs. Model loading, device
selection, and weight format are entirely up to the implementation --
``main_window.py`` never reaches past this interface.
"""

from abc import ABC, abstractmethod


class TileClassifier(ABC):
    name: str  # shown in the UI (e.g. a picker, if more than one is registered)
    categories: list[str]  # predicted label vocabulary, in a stable order

    @abstractmethod
    def predict(self, paths):
        """Return a list of (label, confidence) pairs, one per path in *paths*."""
