"""EfficientNet-B0 with a stem modified for 2-channel (brightfield +
fluorescence) crops, stride=1 so 64x64 spatial resolution survives the
first conv. Shared by both training (`training/supervised.py`) and
inference (`classifiers/yeast_efficientnet.py`) so the two can never
architecturally drift apart -- a training run producing weights the
inference side can't load would otherwise be a silent, late-discovered
bug.

Ported from NN_workflow/yeastVIC.py's `get_modified_efficientnet`, minus
the `num_classes=None` -> `nn.Identity()` branch that function also
supported for VICReg's headless feature extraction -- not needed until
VICReg pretraining itself is formalized (see training/__init__.py).
"""


def build_yeast_efficientnet(num_classes: int, pretrained: bool = True):
    """`pretrained=True` starts from ImageNet weights (only the stem's
    first conv is hand-modified for 2 input channels; see below) --
    appropriate for a cold start with no existing checkpoint. Training
    that warm-starts from an existing checkpoint should build with
    `pretrained=False` and then call `.load_state_dict(...)` itself, since
    the ImageNet init would just be immediately overwritten anyway."""
    import torch.nn as nn
    import torchvision.models as models

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=(1, 1),  # was (2, 2) -- keeps 64x64 spatial resolution intact
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    if pretrained:
        # The pretrained stem expects 3 RGB channels; ours has 2. Average
        # the RGB filters into a single filter and duplicate it across
        # both new channels, rescaling by 3/2 so the expected activation
        # magnitude (mean_weight * num_input_channels) matches the
        # original stem.
        import torch

        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, 2, 1, 1) * (3 / 2))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
