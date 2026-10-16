from yeastprep.core.channels import ChannelSelection, infer_channel_selection


def test_infer_by_name_hints():
    channels_meta = [
        {"name": "White Light - Brightfield"},
        {"name": "EPI - 405 (DAPI)"},
    ]
    result = infer_channel_selection(channels_meta)
    assert result == ChannelSelection(brightfield=0, projection=1)


def test_infer_by_wavelength_when_name_missing():
    channels_meta = [
        {"name": "Trans"},
        {"name": "Channel 2", "emission_wavelength": 460.0},
    ]
    result = infer_channel_selection(channels_meta)
    assert result == ChannelSelection(brightfield=0, projection=1)


def test_infer_returns_none_for_empty_metadata():
    assert infer_channel_selection([]) is None
    assert infer_channel_selection(None) is None


def test_infer_returns_none_when_ambiguous():
    channels_meta = [{"name": "Channel 1"}, {"name": "Channel 2"}]
    assert infer_channel_selection(channels_meta) is None


def test_infer_returns_none_for_single_channel():
    assert infer_channel_selection([{"name": "Brightfield"}]) is None


def test_short_name_hints_match_whole_words_only():
    """"bf" must not pick a BFP fluorescence channel as brightfield."""
    from yeastprep.core.channels import is_brightfield_name

    for name in ("BF", "bf_100ms", "TL BF", "DIC2", "Trans", "Brightfield100", "White Light"):
        assert is_brightfield_name(name), name
    for name in ("BFP", "mTagBFP2", "EGFP", "Transfected-GFP"):
        assert not is_brightfield_name(name), name

    channels = [{"name": "mTagBFP2"}, {"name": "BF"}, {"name": "DAPI-405"}]
    assert infer_channel_selection(channels) == ChannelSelection(brightfield=1, projection=2)
