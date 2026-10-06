import pytest
import torch
from torch.nn import functional as F

from krea2_slider_node.regional_attention import build_attention_masks, build_region_owners


def masks():
    first = torch.zeros(1, 8, 8)
    first[:, 4:, :4] = 1
    second = torch.zeros_like(first)
    second[:, :, 4:] = 1
    return [first, second]


def test_owners_cover_background_and_overlaps_prefer_first():
    owners = build_region_owners(masks(), (4, 4))
    assert owners.shape == (16,)
    assert owners.tolist() == [2, 2, 1, 1] * 2 + [0, 0, 1, 1] * 2
    overlap = masks()
    overlap[1][:, 4:, 2:] = 1
    assert build_region_owners(overlap, (4, 4)).reshape(4, 4)[3, 1] == 0


def test_full_canvas_and_four_edge_regions():
    full = build_region_owners([torch.ones(1,8,8)], (4,4))
    assert (full == 0).all()
    corners = []
    for y,x in [(0,0),(0,4),(4,0),(4,4)]:
        mask = torch.zeros(1,8,8); mask[:,y:y+4,x:x+4] = 1
        corners.append(mask)
    owners = build_region_owners(corners,(2,2))
    assert owners.tolist() == [0,1,2,3]
    joint, text = build_attention_masks([(i,i+1) for i in range(6)],owners)
    assert joint.any(dim=1).all() and text.any(dim=1).all()
    for index in range(4):
        assert joint[6+index,1+index]
        assert not joint[6+index,6:][torch.arange(4) != index].any()


@pytest.mark.parametrize('kind', ['empty', 'tiny', 'shadowed', 'nan', 'range', 'batch', 'size'])
def test_invalid_regions_fail_before_forward(kind):
    region_masks = masks()
    if kind == 'empty': region_masks[0].zero_()
    elif kind == 'tiny':
        region_masks[0].zero_(); region_masks[0][0, 7, 0] = 1
    elif kind == 'shadowed': region_masks[1] = region_masks[0].clone()
    elif kind == 'nan': region_masks[0][0, 0, 0] = float('nan')
    elif kind == 'range': region_masks[0][0, 0, 0] = 2
    elif kind == 'batch': region_masks[0] = region_masks[0].repeat(2, 1, 1)
    elif kind == 'size': region_masks[1] = torch.ones(1, 4, 4)
    with pytest.raises(ValueError): build_region_owners(region_masks, (4, 4))


def test_strict_attention_blocks_text_and_background_relays():
    segments = [(0, 2), (2, 4), (4, 6), (6, 8)]
    owners = torch.tensor([0, 0, 1, 2])
    joint, text = build_attention_masks(segments, owners, 'strict')
    assert joint.shape == (12, 12) and text.shape == (8, 8)
    assert joint.all(dim=1).sum() == 0
    assert joint.any(dim=1).all() and text.any(dim=1).all()
    assert not joint[0:2, 2:].any()  # base cannot collect and relay region information
    assert not joint[2:4, 10:].any()  # female text cannot see male/background images
    assert not joint[10:, 2:4].any()
    assert not joint[11, 8:11].any()  # background cannot relay region images
    assert joint[8:10, 2:4].all()
    assert joint[8:10, :2].all()
    assert not text[4:6, 2:4].any()


def rollout(embedding, allow):
    result = embedding.clone()
    for _ in range(3):
        for _ in range(3):
            result = result + 0.1 * F.scaled_dot_product_attention(
                result[None, None], result[None, None], result[None, None],
                attn_mask=allow[None, None])[0, 0]
    return result


def test_prompt_perturbation_stays_local_across_layers_and_steps():
    torch.manual_seed(19)
    segments = [(0, 2), (2, 4), (4, 6), (6, 8)]
    owners = torch.tensor([0, 0, 1, 2])
    allow, _ = build_attention_masks(segments, owners, 'strict')
    embedding = torch.randn(12, 8)
    changed = embedding.clone(); changed[2:4] += 3
    before, after = rollout(embedding, allow), rollout(changed, allow)
    outside = [0, 1, 4, 5, 6, 7, 10, 11]
    torch.testing.assert_close(before[outside], after[outside], atol=1e-5, rtol=0)
    assert (before[8:10] - after[8:10]).abs().max() > 1e-4
    # The source's base/background relay routes would defeat this invariant.
    relay = allow.clone(); relay[:2, 8:] = True; relay[11, 8:] = True
    assert (rollout(embedding, relay)[outside] - rollout(changed, relay)[outside]).abs().max() > 1e-4


def test_segments_must_be_contiguous_and_match_owner_count():
    with pytest.raises(ValueError):
        build_attention_masks([(0, 1), (2, 3), (3, 4)], torch.tensor([0, 1]), 'strict')
    with pytest.raises(ValueError):
        build_attention_masks([(0, 1), (1, 2), (2, 3)], torch.tensor([0, 2]), 'strict')
