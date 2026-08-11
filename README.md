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

### Local path dependencies

`pyvistra`, `resolvde`, and `psfkit` are pulled in as editable local path
dependencies (see `[tool.uv.sources]` in `pyproject.toml`) rather than from
PyPI or a uv workspace. Before `uv sync` will succeed on a new machine, clone
them to matching paths:

```bash
mkdir -p ~/CustomPythonPackages
git clone https://github.com/delnatan/pyvistra ~/CustomPythonPackages/pyvistra
# resolvde and psfkit currently have no git remote — copy them over manually
```

If you clone/copy them elsewhere, update the paths in `[tool.uv.sources]`
in `pyproject.toml` to match.

### GPU / PyTorch

`pyproject.toml` pins `torch` to a CUDA 13.2 build for Linux via the
`pytorch-cu132` index. Adjust `[[tool.uv.index]]` if the target machine has a
different CUDA version or no GPU.

## Running

```bash
uv run tiled_viewer
uv run yeastprep
uv run pytest
```
