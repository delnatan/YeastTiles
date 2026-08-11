"""Registry of available auto-annotators, keyed by display name.

Each entry is a zero-arg factory rather than a ready instance, so
listing/picking a classifier never imports its (possibly heavy, e.g.
torch) dependencies -- only actually selecting one does.

To add another network (a different classification task), drop a new
module next to ``yeast_efficientnet.py`` implementing ``TileClassifier``
(see ``base.py``), give it its own subfolder under ``weights/``, and
register a factory for it below.
"""

CLASSIFIERS = {}


def _register_yeast_efficientnet():
    from .yeast_efficientnet import YeastEfficientNetClassifier

    return YeastEfficientNetClassifier()


CLASSIFIERS["Yeast (bf + fluorescence + mask)"] = _register_yeast_efficientnet
