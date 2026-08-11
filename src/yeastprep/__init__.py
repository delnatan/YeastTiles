"""yeastprep: raw-stack visual diagnostics and field-flattening for yeast FOVs.

Upstream of ``tileclass`` in the pipeline described in design.md: takes raw
multi-channel Z-stacks (brightfield + a target fluorescence channel) and
produces a flattened focal-plane brightfield image plus a sum-projected
target-channel image, ready for cellpose segmentation and tile cropping.
"""
