"""Before/after preview for the Deconvolve stage: target channel only
(deconvolution never touches brightfield -- see design.md).

Built on the shared vispy-based `BeforeAfterPreviewPanel` (see
`ui/common/before_after_preview.py`) rather than its own matplotlib figure
-- `set_data` keeps its original 2-argument signature so `deconvolve_page.py`
needs no changes beyond this import.
"""

from ..common.before_after_preview import BeforeAfterPreviewPanel


class DeconvolvePreviewPanel(BeforeAfterPreviewPanel):
    pass
