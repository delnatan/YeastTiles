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
