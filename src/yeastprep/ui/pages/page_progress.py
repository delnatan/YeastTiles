"""Plain data payload for a page's background-work progress -- no Qt, no
behavior. Pages emit this from their `progress_changed` signal; the shell
renders it into that page's sidebar progress bar. Keeping it a bare
dataclass (rather than, say, passing (done, total, message) as three
signal args, or reaching into the page's worker from the shell) is what
lets the shell stay ignorant of *how* a page produces progress -- it only
ever consumes this one small, page-agnostic shape.
"""

from dataclasses import dataclass


@dataclass
class PageProgress:
    active: bool
    done: int = 0
    total: int = 0
    message: str = ""
