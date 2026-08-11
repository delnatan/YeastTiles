"""Top-level pipeline-stage pages hosted by ui/main_window.py's page-list
shell. Each page is a self-contained QWidget: it owns its own folder
panel(s), worker thread(s), and status feedback, and works standalone as
long as the right input files are present on disk (design.md's stage
boundaries) -- pages don't reference each other directly. Any convenience
wiring that legitimately needs to know about more than one page (e.g. the
Segmentation page's "use the Data Reduction output folder" shortcut) is
connected by the shell, the one place that holds references to both.
"""
