import importlib
import importlib.util
import json

import pytest
import torch


def api():
    assert importlib.util.find_spec('krea2_slider_node.region_masks') is not None, 'Region mask implementation is missing'
    return importlib.import_module('krea2_slider_node.region_masks')


def test_default_pair_covers_left_and_right_without_gap_or_overlap():
    first, second = api().create_region_masks(10, 6, '[]')
    assert first.shape == second.shape == (1, 6, 10)
    assert first.dtype == second.dtype == torch.float32
    assert torch.all(first[:, :, :5] == 1) and torch.all(first[:, :, 5:] == 0)
    assert torch.all(second[:, :, :5] == 0) and torch.all(second[:, :, 5:] == 1)
    assert torch.all(first + second == 1)


def test_edited_regions_keep_relative_placement_at_another_resolution():
    saved = json.dumps([{'x': .25, 'y': .25, 'w': .25, 'h': .5},
                        {'x': .75, 'y': 0, 'w': .25, 'h': 1}])
    first, second = api().create_region_masks(16, 8, saved)
    expected = torch.zeros(1, 8, 16)
    expected[:, 2:6, 4:8] = 1
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert second.sum() == 32
    large, _ = api().create_region_masks(32, 16, saved)
    assert large.sum() == 64
    assert torch.all(large[:, 4:12, 8:16] == 1)


def test_outside_rectangle_is_moved_inside_without_shrinking():
    saved = json.dumps([{'x': .9, 'y': -.2, 'w': .5, 'h': .5},
                        {'x': .5, 'y': 0, 'w': .5, 'h': 1}])
    first, _ = api().create_region_masks(8, 8, saved)
    assert first.sum() == 16
    assert torch.all(first[:, :4, 4:] == 1)


@pytest.mark.parametrize('saved', ['{', '{}', '[null,null]',
                                  '[{"x":0,"y":0,"w":NaN,"h":1},{}]',
                                  '[{"x":0,"y":0,"w":1,"h":1}]'])
def test_invalid_saved_geometry_is_reported_instead_of_silently_changing_target(saved):
    with pytest.raises(ValueError, match='region|Region|rectangle|Rectangle'):
        api().create_region_masks(8, 8, saved)


@pytest.mark.parametrize('width,height', [(0, 8), (8, -1), (True, 8), (8.5, 8), (20000, 8)])
def test_invalid_dimensions_are_rejected_before_allocating_masks(width, height):
    with pytest.raises(ValueError, match='width|height|dimension'):
        api().create_region_masks(width, height, '[]')


def test_small_rectangle_at_edge_keeps_at_least_one_pixel():
    saved = json.dumps([{'x': 1, 'y': 1, 'w': .01, 'h': .01},
                        {'x': .5, 'y': 0, 'w': .5, 'h': 1}])
    first, _ = api().create_region_masks(8, 8, saved)
    assert first.sum() == 1 and first[0, -1, -1] == 1
