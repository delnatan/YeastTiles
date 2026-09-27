# YeastTiles

See [design.md](design.md) for the project overview and pipeline.

Two installable packages live under `src/`:

- `tileclass` — tile-grid viewer/classifier (`tiled_viewer` entry point)
- `yeastprep` — raw-image prep pipeline (`yeastprep` / `yeastprep-batch` entry points)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment
management. Clone the repo, then sync the profile that matches the machine:

```bash
uv sync --extra gpu       # NVIDIA GPU workstation
uv sync --extra gpu-amd   # AMD GPU workstation (Linux only)
uv sync --extra lite      # laptop / tablet / anything else
```

- **`gpu`** -- the whole pipeline: `yeastprep` data reduction, denoise,
  deconvolution, segmentation, tile generation, and VICReg/classifier
  training. Pulls the CUDA 13.2 torch build on Linux/Windows.
- **`gpu-amd`** -- the same as `gpu`, on the ROCm 7.2 torch build (Linux
  x86_64 only; see "Installation notes" below).
- **`lite`** -- `tiled_viewer` browsing and annotation, running and
  fine-tuning the classifier, and plotting. Pulls the CPU-only torch build
  on Linux/Windows (a few hundred MB instead of several GB of CUDA
  libraries). A project can consist of nothing but its packed `.tiles`
  containers -- `tiled_viewer` has no notion of FOVs, raw stacks, or a
  `yeastprep` project, so point it at container files directly (see "Using
  tileclass" below). `yeastprep` also launches here: its Classifier Training
  and Classify Tiles pages work, and the Denoise and Segmentation pages say
  which packages they'd need.

The profiles are declared conflicting, so one environment holds one of
them. Pass the same `--extra` on every later `uv sync` -- a bare `uv sync`
removes the extras again. (`uv run` doesn't; it leaves an already-synced
environment alone.)

Tests and the dev group: `uv sync --extra gpu` already includes the `dev`
group (pytest) by default. Notebooks under `notebooks/` need
`--extra notebooks` on top of `gpu`.

### Installation notes

- **macOS (Apple silicon).** Every profile uses the regular PyPI torch,
  which runs on the GPU through Apple's Metal backend (MPS); the profiles
  only differ in which packages come along.
- **Intel Macs, including iMacs with AMD Radeon GPUs, can't run any
  profile.** PyTorch stopped publishing Intel-Mac wheels after 2.2, and
  ROCm doesn't exist on macOS, so there's no torch build to install. They
  can still run the base install (`uv sync` with no extras) to browse and
  hand-annotate tiles in `tiled_viewer`; run classification on another
  machine. Supporting them would take a separate old stack (torch 2.2.2 on
  Python 3.12 or older, against the 3.13 pin), with unreliable MPS on those
  GPUs, so it isn't set up.
- **AMD GPUs (`gpu-amd`) are Linux x86_64 only** -- PyTorch publishes no
  ROCm wheels for Windows or macOS. The machine needs a ROCm-supported
  Radeon/Instinct GPU with AMD's Linux driver installed. ROCm torch drives
  the GPU through the same `torch.cuda` API, so the code is the same as on
  NVIDIA; the training page's device picker just labels it "ROCm".
- **Windows.** `gpu` (NVIDIA) and `lite` both work; there's no AMD GPU
  option.

### Building-block extras

The profiles are bundles of smaller extras, which can still be combined by
hand for an unusual machine:

- `classification` -- torch, pyvistra/qtkit, scikit-learn, plotting.
- `prep` -- cellpose, jssl-denoise, scipy/vispy (Denoise, Segmentation).
- `psf` -- `psfkit`, only for the PSF Calculator tab on the Deconvolve
  page. Deconvolution itself just needs a PSF tiff file.
- `notebooks` -- jupyter.

Without a profile, torch comes from plain PyPI (CUDA 13.0 on Linux
x86_64, CPU-only on Windows). The base install with no extras at all is
just the Qt tile viewer; Auto-Annotate then shows a dialog about the
missing `classification` packages rather than crashing.

### Third-party packages

`jssl-denoise` (`prep`), `pyvistra` and `qtkit` (`classification`), and
`psfkit` (`psf`) are git-URL dependencies in `pyproject.toml` -- `uv sync`
clones them itself. `resolvde` isn't a dependency at all: its
deconvolution code is vendored into `src/yeastprep/core/deconvolution/`
(see that package's docstring).

### GPU / PyTorch

The torch build per profile is set in `[tool.uv.sources]` /
`[[tool.uv.index]]` in `pyproject.toml`: the `pytorch-cu132` index for
`gpu`, `pytorch-rocm72` for `gpu-amd`, `pytorch-cpu` for `lite`, PyPI on
macOS. For a GPU machine whose driver doesn't support CUDA 13.2 or ROCm 7.2,
point that index at a different `download.pytorch.org/whl/cuXXX` or
`.../rocmX.Y` URL and re-run `uv lock`.

This only works through `uv sync` / `uv run` (uv's project workflow) -- `pip
install` doesn't read `[tool.uv.sources]`. `uv.lock` and `.python-version`
are committed, so `uv sync` on a new machine installs Python 3.13 if needed
and reproduces the exact resolved versions, including the pinned git
commits, without re-resolving.

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
the tile viewer and Classify Tiles use.

## Tests

```bash
uv run pytest
```
