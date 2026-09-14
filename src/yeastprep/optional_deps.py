"""Checks for the packages that only some install profiles have (see the
extras in pyproject.toml). A `classification`-only install has no
cellpose/jssl-denoise, so modules that need those import them inside the
function that uses them (via `require`, for a message that names the
extra to install), and ui/main_window.py uses `missing` to show an
"install the `prep` extra" placeholder instead of a page it can't build.
"""

import importlib
import importlib.util


def missing(*packages: str) -> list[str]:
    """The subset of top-level `packages` that aren't installed. Doesn't
    import them, so it's cheap to call for torch-sized packages."""
    return [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]


def require(module: str, extra: str):
    """Import `module`, re-raising a missing package as an ImportError
    that says which extra provides it."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{exc.name or module!r} isn't installed; it comes with the "
            f"'{extra}' extra: pip install 'tileclass[{extra}]'"
        ) from exc
