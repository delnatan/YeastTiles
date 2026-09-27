"""Pages hosted by ui/main_window.py. Each page is a self-contained QWidget
that owns its worker thread(s) and status feedback, reads the project
through the shared ProjectTreePanel, and works as long as its input files
are on disk -- pages don't reference each other directly. The little
cross-page wiring there is (e.g. a freshly trained checkpoint defaulting
Classify Tiles' pickers) is connected by the shell.
"""
