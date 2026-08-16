"""Before/after preview for the Denoise stage: just the channel currently
selected in DenoiseParamsPanel, since brightfield and target/fluorescence
are denoised one at a time with different networks -- showing both at once
would mix results from two unrelated runs.

Built on the shared vispy-based `BeforeAfterPreviewPanel` (see
`ui/common/before_after_preview.py`) rather than its own matplotlib figure
-- `set_data`/`clear` keep their original signatures so `denoise_page.py`
needs no changes beyond this import.
"""

from ..common.before_after_preview import BeforeAfterPreviewPanel


class DenoisePreviewPanel(BeforeAfterPreviewPanel):
    pass
