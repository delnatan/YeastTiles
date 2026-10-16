"""`core.model_cache.ModelCache`: reuse a loaded model until its file
changes, and keep only the few most recently used."""

import os

from yeastprep.core.model_cache import ModelCache


def test_reuses_until_the_file_is_rewritten(tmp_path):
    checkpoint = tmp_path / "denoise_brightfield.pt"
    checkpoint.write_bytes(b"v1")
    cache = ModelCache()
    loads = []

    def load():
        loads.append(checkpoint.read_bytes())
        return object()

    first = cache.get(checkpoint, "cpu", load)
    assert cache.get(checkpoint, "cpu", load) is first
    assert loads == [b"v1"]

    checkpoint.write_bytes(b"v2, retrained")
    st = checkpoint.stat()
    os.utime(checkpoint, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert cache.get(checkpoint, "cpu", load) is not first
    assert loads == [b"v1", b"v2, retrained"]

    assert cache.get(checkpoint, "mps", load) is not first  # per device
    assert len(loads) == 3


def test_keeps_only_the_most_recently_used(tmp_path):
    cache = ModelCache(max_entries=2)
    loads = []

    def loader(name):
        def load():
            loads.append(name)
            return name

        return load

    for name in ("a", "b", "a", "c", "a", "b"):
        cache.get(name, "cpu", loader(name))
    # "a" stayed in use, "b" was evicted by "c" and loaded again.
    assert loads == ["a", "b", "c", "b"]
