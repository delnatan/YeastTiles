# YeastTiles

See [design.md](design.md) for the project overview and pipeline.

Two installable packages live under `src/`:

- `tileclass` — tile-grid viewer/classifier (`tiled_viewer` entry point)
- `yeastprep` — raw-image prep pipeline (`yeastprep` / `yeastprep-batch` entry points)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync
```

### Third-party packages

`jssl-denoise`, `pyvistra`, and `psfkit` are listed directly as git-URL
dependencies in `pyproject.toml`'s `dependencies` list -- `uv sync` clones
them itself; no manual cloning or local path setup needed. `resolvde` isn't
a dependency at all: its deconvolution code is vendored directly into
`src/yeastprep/core/deconvolution/` (see that package's docstring).

### GPU / PyTorch

`pyproject.toml` pins `torch`/`torchvision` to a CUDA 13.2 build on Linux
(and Windows) via the `pytorch-cu132` index in `[tool.uv.sources]` /
`[[tool.uv.index]]`. Adjust `[[tool.uv.index]]` if the target machine has a
different CUDA version or no GPU.

This only works through `uv sync` (or `uv lock`/`uv add`) -- `[tool.uv.sources]`
is a uv-project-workflow feature, not something `pip install` or
`uv pip install` reads, so either of those would fall back to a plain PyPI
`torch` wheel (CPU-only on Linux) instead of the pinned CUDA build. Since
`uv.lock` is committed, plain `uv sync` on a new machine reproduces the
exact same resolved versions (including the CUDA wheel and the pinned git
commits above) without needing to re-resolve anything.

## Data model

A "project" is just a folder that holds raw multi-channel Z-stacks (`.ims`/
`.czi`/`.nd2`). There's no separate raw-input picker -- you point `yeastprep`
at that folder and it creates numbered stage subfolders in place, alongside
the raw files:

```
<project>/
  <stem>.ims / .czi / .nd2   raw 3D stacks (untouched by yeastprep)
  01_reduced/       <stem>.tiff   2-channel 2D: [brightfield, target]
  02_denoised/      <stem>.tiff   optional
  03_deconvolved/   <stem>.tiff   optional
  05_tiles/         <fov>_cell#####.tif + tile_index.csv
  .yeastprep_project.json   per-project params, run history, source-stage choices
```

Pipeline, in order (see [design.md](design.md) for the full rationale):

1. **Raw -> Reduced (01_reduced)**: the brightfield channel is flattened to
   its best-focus 2D plane; a target/fluorescence channel is sum-projected.
   Both are saved as one 2-channel tiff per field of view -- this is the
   step that collapses the large raw footprint down to something small
   enough to work with directly.
2. **Denoise (02_denoised)** -- optional, jssl-denoise.
3. **Deconvolve (03_deconvolved)** -- optional, Poisson-ML deconvolution of
   the target channel only (brightfield has no PSF model). Reads from
   02_denoised if that ran, else 01_reduced.
4. **Segment**: Cellpose masks each cell in the brightfield channel of
   whichever 2D stage is currently the "source" (see below) -- masks are
   cellpose's own `_seg.npy` sidecars, written next to the source image
   rather than into a folder of their own, so the real Cellpose GUI can
   open that folder directly for manual correction.
5. **Tile (05_tiles)**: crops every segmented cell into a fixed-size
   3-channel tile (brightfield, target, mask). **Tiles are the project's
   primary data** -- what gets pooled across experiments, annotated, and
   used for classification; everything upstream exists to produce them
   reproducibly, not as an end in itself.

Since Denoise/Deconvolve are optional, "the source stage" for Segmentation
and Tile Generation is resolved automatically (most-downstream stage that
has output wins: deconvolved > denoised > reduced), or pinned explicitly
via the tree panel's source dropdown -- persisted per-project.

**Archiving raw/intermediate data**: raw stacks are large and are expected
to eventually move to a storage server and get deleted from the local
disk once a stage has consumed them (the same will likely happen to
01-03 once tiling is done). The app is built around that: freshness is
inferred live from what's on disk, and a stage whose producer folder has
been emptied out shows as a distinct "archived" status (blue) rather than
a false "stale" (amber) -- it just means "can't verify, presumed fine,"
not "something's wrong." Nothing needs to be marked or configured for
this; it self-heals if the archive is ever remounted.

## Using yeastprep (processing pipeline)

```bash
uv run yeastprep [path-to-project-folder]
```

The window has three parts: a pipeline breadcrumb across the top (stage ->
stage status at a glance), a sidebar with the page list and the project
tree, and the current page filling the rest.

The project tree lists every stage's files. **Click a file** to select it --
a "Selection" panel under the tree then lists exactly which tasks apply to
it right now (e.g. a 01_reduced file might offer *Denoise this file*,
*Deconvolve this file*, *Segment this file*, *Preview*), computed from the
project's actual current state rather than a fixed rule. **Click one of
those actions** to jump to the page that handles it, with the file already
loaded. Checkboxes in the tree control which files a page's batch button
processes; batch buttons enable only once there's something valid to run.

## Using tileclass (tile viewer / annotation)

```bash
uv run tiled_viewer <tiles-folder> [<more-folders>...] [--fov NAME ...]
```

Also reachable from yeastprep's Tile Generation page ("Open in Tile
Viewer"). Browses cropped cell tiles in a grid, lets you annotate/label
them, and fine-tune an EfficientNet classifier (frozen-backbone probe, then
full unfreeze) on the labeled set -- starting from whatever's currently
deployed, or an ImageNet-pretrained stem for a first run. Pass multiple
folders to pool tiles from several experiments into one session. Note: the
self-supervised VICReg embedding-pretraining step described in design.md
isn't wired into this app yet -- classifier training/fine-tuning doesn't
depend on it.

## Tests

```bash
uv run pytest
```
