"""Manual area metadata for Krea2's 3D latent masked conditioning."""
import pytest
import torch

from krea2_slider_node.region_area import (apply_area_to_conditioning_pair,
                                           latent_area_from_mask)


@pytest.mark.parametrize("latent_shape", [(1, 16, 128, 128), (1, 16, 1, 128, 128)])
def test_mask_bounds_become_spatial_latent_area(latent_shape):
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 376:, :472] = 1

    assert latent_area_from_mask(mask, {"samples": torch.zeros(latent_shape)}) == (81, 59, 47, 0)


def test_pair_area_keeps_hook_and_mask_without_mutating_originals():
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 376:, :472] = 1
    hooks = object()
    positive = [[torch.zeros(1), {"mask": mask, "hooks": hooks, "set_area_to_bounds": True}]]
    negative = [[torch.zeros(1), {"mask": mask, "hooks": hooks, "set_area_to_bounds": True}]]
    latent = {"samples": torch.zeros(1, 16, 1, 128, 128)}

    pos_out, neg_out = apply_area_to_conditioning_pair(positive, negative, mask, latent)

    for output in (pos_out, neg_out):
        assert output[0][1]["area"] == (81, 59, 47, 0)
        assert output[0][1]["set_area_to_bounds"] is False
        assert output[0][1]["mask"] is mask
        assert output[0][1]["hooks"] is hooks
    assert "area" not in positive[0][1] and positive[0][1]["set_area_to_bounds"] is True
    assert "area" not in negative[0][1] and negative[0][1]["set_area_to_bounds"] is True


def test_empty_mask_is_rejected_before_sampling():
    mask = torch.zeros(1, 32, 32)

    with pytest.raises(ValueError, match="nonzero"):
        latent_area_from_mask(mask, {"samples": torch.zeros(1, 16, 1, 4, 4)})


def test_unbatched_mask_is_rejected_for_3d_latent_sampling():
    with pytest.raises(ValueError, match="batched"):
        latent_area_from_mask(torch.ones(32, 32), {"samples": torch.zeros(1, 16, 1, 4, 4)})


def test_conditioning_mask_must_match_the_area_source_mask():
    mask = torch.ones(1, 32, 32)
    other = torch.zeros(1, 32, 32)
    positive = [[torch.zeros(1), {"mask": other}]]
    negative = [[torch.zeros(1), {"mask": mask}]]

    with pytest.raises(ValueError, match="same MASK"):
        apply_area_to_conditioning_pair(positive, negative, mask,
                                        {"samples": torch.zeros(1, 16, 1, 4, 4)})
