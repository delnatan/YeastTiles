# YeastTiles

See [design.md](design.md) for the project overview and pipeline.

Two installable packages live under `src/`:

- `tileclass` — tile-grid viewer/classifier (`tiled_viewer` entry point)
- `yeastprep` — raw-image prep pipeline (`yeastprep` / `yeastprep-batch` entry points)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

The two packages have very different weight, and installs are split to match:

- **Just classifying/annotating tiles** (`tileclass` / `tiled_viewer`) --
  light enough for a Windows tablet or any modest laptop. The base install
  is just the Qt viewer (no torch, no GPU-segmentation stack); add the
  `classification` extra for running or fine-tuning the classifier:

  ```bash
  uv sync --extra classification
  ```

  A project can consist of nothing but its packed `.tiles` containers --
  `tiled_viewer` has no notion of FOVs, raw stacks, or a `yeastprep`
  project at all, so there's nothing to remove: point it at container
  files directly (see "Using tileclass" below) and it works the same
  whether or not any upstream FOV data exists on the machine. This is the
  intended workflow
  for fixing annotations and running the classifier day-to-day, keeping
  the (much heavier) VICReg/full-network training for a more capable
  machine.

- **Full pipeline, raw stacks through tiles** (`yeastprep`) -- needs both
  `prep` (cellpose, jssl-denoise) and `classification` (torch, pyvistra,
  qtkit):

  ```bash
  uv sync --extra classification --extra prep
  ```

  `yeastprep` launches with just `classification`: its Classifier Training
  and Classify Tiles pages work, and the Denoise and Segmentation pages show
  which packages to install instead. Running it
  without `classification` shows a dialog naming the missing packages. `psf`
  is a further, separate extra -- it's only `psfkit`, used by the PSF
  Calculator convenience tab on the Deconvolve page. Deconvolution itself
  just needs a PSF tiff file, so skipping `psf` doesn't block it; that tab
  just shows a friendly message instead of computing a PSF for you.

  ```bash
  uv sync --extra classification --extra prep --extra psf   # + PSF Calculator
  ```

  Equivalently, `uv sync --extra full` bundles `classification` + `prep` + `psf`.

- **Development / running the test suite** needs both extras plus the dev
  group:

  ```bash
  uv sync --extra classification --extra prep --extra psf --group dev
  ```

- **Notebooks** under `notebooks/` (field-flattening, cookie-cutting) need
  `prep` (pyvistra/cellpose) plus:

  ```bash
  uv sync --extra notebooks
  ```

Within `tiled_viewer` itself, Auto-Annotate is what actually imports torch -- if the `classification` extra isn't installed, that
surfaces as a dialog rather than a crash, so the base install stays usable
purely for browsing/annotating tiles even without deciding on `classification` up front.

### Third-party packages

`jssl-denoise` (`prep` extra), `pyvistra` and `qtkit` (`classification` extra),
and `psfkit` (`psf` extra) are
listed directly as git-URL dependencies in `pyproject.toml` -- `uv sync`
clones them itself; no manual cloning or local path setup needed. `resolvde` isn't
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
  05_tiles/         <fov>.tiles (packed cell crops) + tile_index.csv
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
   3-channel tile (brightfield, target, mask), packed into one compressed
   container per FOV (`05_tiles/<fov>.tiles`, see
   `tileclass/tile_container.py`) so each FOV keeps its own annotation
   sidecar file without needing thousands of loose per-cell files on disk.
   **Tiles are the project's primary data** -- what gets pooled
   across experiments, annotated, and used for classification; everything
   upstream exists to produce them
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

The window has three parts: a navigation strip across the top, a sidebar
with the project tree, and the current page filling the rest. The strip has
one chip per pipeline stage (colored by that stage's status), followed by
the Classifier Training and Classify Tiles pages. Click a chip to open its
page; the current page is underlined, and a thin bar under a chip shows a
batch or training run still going on that page.

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
uv run tiled_viewer <fov>.tiles [<more>.tiles ...] [--fov NAME ...]
```

Also reachable from yeastprep's Tile Generation page ("Open in Tile
Viewer"). Browses the cell tiles of one or more `.tiles` containers in a
grid (one container per FOV; tiles are shown by FOV and cell index), and
annotates them. Each container keeps its own annotation sidecar
(`<fov>.txt` next to it), so passing several containers pools tiles from
several experiments into one session.

Annotating: click / Shift-click / Ctrl-click (or Ctrl+A) to select tiles,
then press a number key to apply that category -- the legend under the
toolbar lists them (Annotate > Manage Categories... edits the list). T picks
a category from a list instead, Delete clears it. Annotate > Auto-Annotate
Page runs the deployed classifier on the current page's un-annotated tiles;
its predictions show a confidence and a dashed outline until you accept them
(A) or relabel them. Group by Category (G) sorts the page so the least
confident predictions come first; Clear AI Predictions drops every
unconfirmed prediction so a newer model can redo them.

Training the classifier (supervised fine-tuning, and VICReg backbone
pretraining) happens on yeastprep's Classifier Training page, and running it
over whole projects on its Classify Tiles page. Both pages share one
classification pool of projects, kept between sessions (select a project's
05 · Tiles entry in the tree and choose "Add Project to Classification
Pool", or use "Add project..."), and both use only human-set or accepted
annotations as ground truth. A training run saves to the first pooled
project's `06_classifier/` folder; "Deploy Latest" then makes it the model
the tile viewer and Classify Tiles use. An already-exported
project with loose per-cell tifs from before the packed-container format can
be converted with `uv run yeastprep-pack-tiles <project_root>`.

## Tests

```bash
uv run pytest
```
