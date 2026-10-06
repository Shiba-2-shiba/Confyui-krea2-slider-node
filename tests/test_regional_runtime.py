from types import SimpleNamespace
import sys

import pytest
import torch

from krea2_slider_node.regional_runtime import (
    CURRENT_REGIONAL_CALL, RegionSpec, RegionalAttentionWrapper,
    compose_regional_conditioning, merge_attention_masks,
)


def conditioning(length=2, metadata=None):
    return [[torch.randn(1, length, 8), metadata or {}]]


def setup_bundle():
    first = torch.zeros(1, 8, 8); first[:, 4:, :4] = 1
    second = torch.zeros(1, 8, 8); second[:, :, 4:] = 1
    regions = [RegionSpec(conditioning(), first), RegionSpec(conditioning(), second)]
    output, bundle = compose_regional_conditioning(conditioning(), conditioning(), regions,
                                                  feature_dim=8)
    return output[0][0], bundle


def test_compose_preserves_segments_strips_padding_and_snapshots_masks():
    region_mask = torch.ones(1, 8, 8)
    base = conditioning(3, {'attention_mask': torch.tensor([[1, 0, 1]])})
    output, bundle = compose_regional_conditioning(base, conditioning(),
                          [RegionSpec(conditioning(), region_mask)], feature_dim=8)
    assert bundle.segments == ((0, 2), (2, 4), (4, 6))
    assert output[0][0].shape == (1, 6, 8)
    assert 'attention_mask' not in output[0][1]
    region_mask.zero_()
    assert bundle.masks[0].sum() == 64


@pytest.mark.parametrize('case', ['metadata', 'entries', 'features', 'tokens', 'video_lora'])
def test_invalid_conditioning_stops_at_compose(case):
    base, background = conditioning(), conditioning()
    region = RegionSpec(conditioning(), torch.ones(1, 8, 8))
    kwargs = {'feature_dim': 8}
    if case == 'metadata': base[0][1]['hooks'] = object()
    elif case == 'entries': base += conditioning()
    elif case == 'features': base[0][0] = torch.ones(1, 2, 9)
    elif case == 'tokens': kwargs['max_tokens'] = 5
    elif case == 'video_lora': region = RegionSpec(conditioning(), region.mask, loras=(object(),))
    with pytest.raises(ValueError): compose_regional_conditioning(base, background, [region], **kwargs)


class Executor:
    class_obj = SimpleNamespace(patch=2, txtlayers=3)

    def __init__(self, action): self.action = action

    def __call__(self, x, ts, context, *args, transformer_options=None, **kwargs):
        return self.action(x, context, transformer_options)


def attention_probe(options, batch, text_length=8, image_length=16, previous_mask=None):
    q = torch.randn(batch, 2, text_length + image_length, 4)
    captured = {}
    def attention(q, k, v, heads, mask=None, **kwargs):
        captured['mask'] = mask
        return q
    text = q[:, :, :text_length]
    options['optimized_attention_override'](attention, text, text, text, 2, skip_reshape=True)
    options['optimized_attention_override'](attention, q, q, q, 2, mask=previous_mask,
                                            skip_reshape=True)
    return captured.get('mask')  # A native backend replacement does not call this probe backend.


def test_wrapper_applies_mask_and_leaves_same_length_negative_clean():
    context, bundle = setup_bundle(); wrapper = RegionalAttentionWrapper(bundle)
    x = torch.zeros(1, 16, 1, 8, 8)
    options = {'cond_or_uncond': [0]}
    result = wrapper(Executor(lambda x,c,o: attention_probe(o, 1)), x, torch.ones(1),
                     context, transformer_options=options)
    assert torch.isneginf(result[0, 0, 0, 8])
    assert 'optimized_attention_override' not in options
    assert CURRENT_REGIONAL_CALL.get() is None
    clean = wrapper(Executor(lambda x,c,o: o), x, torch.ones(1), context,
                    transformer_options={'cond_or_uncond': [1]})
    assert 'optimized_attention_override' not in clean


def test_native_positional_forward_options_are_copied():
    context, bundle = setup_bundle()
    original = {'cond_or_uncond':[0]}
    class PositionalExecutor(Executor):
        def __call__(self,x,ts,context,attention_mask,ref_latents,transformer_options):
            assert attention_mask is None and ref_latents is None
            transformer_options['native_block_index'] = 2
            return attention_probe(transformer_options,1)
    result = RegionalAttentionWrapper(bundle)(PositionalExecutor(None),torch.zeros(1,16,8,8),
                                              torch.ones(1),context,None,None,original)
    assert torch.isneginf(result[0,0,0,8])
    assert original == {'cond_or_uncond':[0]}


def test_mixed_cfg_rows_keep_unconditional_mask_unrestricted():
    context, bundle = setup_bundle()
    wrapper = RegionalAttentionWrapper(bundle)
    out = wrapper(Executor(lambda x,c,o: attention_probe(o, 2)), torch.zeros(2,16,8,8),
                  torch.ones(2), context.repeat(2,1,1), transformer_options={'cond_or_uncond':[0,1]})
    assert torch.isneginf(out[0, 0, 0, 8])
    assert (out[1] == 0).all()


def test_exception_restores_parent_context_and_does_not_modify_options():
    context, bundle = setup_bundle(); wrapper = RegionalAttentionWrapper(bundle)
    original = {'cond_or_uncond':[0]}
    sentinel = object(); token = CURRENT_REGIONAL_CALL.set(sentinel)
    def failure(x,c,o):
        assert CURRENT_REGIONAL_CALL.get() is not sentinel
        raise RuntimeError('deliberate')
    try:
        with pytest.raises(RuntimeError, match='deliberate'):
            wrapper(Executor(failure), torch.zeros(1,16,8,8),torch.ones(1),context,
                    transformer_options=original)
        assert CURRENT_REGIONAL_CALL.get() is sentinel
        assert original == {'cond_or_uncond':[0]}
    finally: CURRENT_REGIONAL_CALL.reset(token)


def test_existing_bool_and_additive_masks_are_intersected():
    regional = torch.tensor([[True,False],[True,True]])[None,None]
    existing = torch.tensor([[True,True],[False,True]])
    result = merge_attention_masks(regional, existing, torch.float32, batch=1, heads=2)
    assert torch.isneginf(result[0,0,0,1]) and torch.isneginf(result[0,0,1,0])
    bias = torch.tensor([[2.,0.],[-float('inf'),3.]])
    result = merge_attention_masks(regional,bias,torch.float32,batch=1,heads=2)
    assert result[0,0,0,0] == 2 and result[0,0,1,1] == 3
    with pytest.raises(ValueError, match='key'):
        merge_attention_masks(regional,torch.zeros(2,2,dtype=torch.bool),torch.float32,1,2)


def test_previous_override_cannot_silently_discard_or_bypass_regional_mask():
    context,bundle=setup_bundle(); wrapper=RegionalAttentionWrapper(bundle)
    def discard(func,q,k,v,heads,mask=None,**kw): return func(q,k,v,heads,mask=None,**kw)
    result=wrapper(Executor(lambda x,c,o:attention_probe(o,1)),torch.zeros(1,16,8,8),
                   torch.ones(1),context,transformer_options={'optimized_attention_override':discard})
    assert torch.isneginf(result[0,0,0,8])
    def bypass(func,q,k,v,heads,**kw): return q
    with pytest.raises(RuntimeError,match='bypass'):
        wrapper(Executor(lambda x,c,o:attention_probe(o,1)),torch.zeros(1,16,8,8),
                torch.ones(1),context,transformer_options={'optimized_attention_override':bypass})


def test_previous_override_preserves_bias_without_doubling_it():
    context, bundle = setup_bundle()
    def passthrough(func, q, k, v, heads, mask=None, **kw):
        return func(q, k, v, heads, mask=mask, **kw)
    result = RegionalAttentionWrapper(bundle)(
        Executor(lambda x,c,o:attention_probe(o,1,previous_mask=torch.full((24,24),2.))),
        torch.zeros(1,16,8,8), torch.ones(1), context,
        transformer_options={'optimized_attention_override':passthrough})
    assert result[0,0,0,0] == 2


def test_mask_cannot_expand_attention_batch():
    with pytest.raises(ValueError):
        merge_attention_masks(torch.ones(1,1,2,2,dtype=torch.bool),
                              torch.ones(3,1,2,2),torch.float32,1,2)


def test_joint_only_engine_is_rejected_without_text_fusion_mask():
    context, bundle = setup_bundle()
    def joint_only(x, c, options):
        q = torch.randn(1,2,24,4)
        return options['optimized_attention_override'](lambda q,k,v,h,**kw:q, q,q,q,2)
    with pytest.raises(RuntimeError, match='text'):
        RegionalAttentionWrapper(bundle)(Executor(joint_only),torch.zeros(1,16,8,8),
                                          torch.ones(1), context)


def test_additive_mask_is_reused_across_layers_and_forwards():
    context, bundle = setup_bundle()
    wrapper = RegionalAttentionWrapper(bundle)
    executor = Executor(lambda x,c,o:attention_probe(o,1))
    first = wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context)
    second = wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context)
    assert first.data_ptr() == second.data_ptr()
    wrapper.reset()
    assert not wrapper.cache


def test_known_native_backend_replacement_receives_regional_mask(monkeypatch):
    # Unit adapter for the core setter's closure contract; real-core test is separate.
    class NativeSetter:
        def set_model_optimized_attention(self, optimized_attention):
            def optimized_attention_override(_, *args, **kwargs):
                return optimized_attention(*args, **kwargs)
            self.override = optimized_attention_override
    monkeypatch.setitem(sys.modules,'comfy.model_patcher',SimpleNamespace(ModelPatcher=NativeSetter))
    setter = NativeSetter()
    captured = []
    def backend(q,k,v,heads,mask=None,**kw):
        captured.append(mask)
        return q
    monkeypatch.setitem(sys.modules,'comfy.ldm.modules.attention',SimpleNamespace(attention_pytorch=backend))
    setter.set_model_optimized_attention(backend)
    context,bundle = setup_bundle()
    wrapper = RegionalAttentionWrapper(bundle)
    wrapper(Executor(lambda x,c,o:attention_probe(o,1)),torch.zeros(1,16,8,8),torch.ones(1),context,
            transformer_options={'optimized_attention_override':setter.override})
    assert len(captured) == 2
    assert torch.isneginf(captured[-1][0,0,0,8])
    setter.set_model_optimized_attention(lambda q,k,v,heads,**kw:q)
    with pytest.raises(RuntimeError, match='bypass'):
        wrapper(Executor(lambda x,c,o:attention_probe(o,1)),torch.zeros(1,16,8,8),torch.ones(1),context,
                transformer_options={'optimized_attention_override':setter.override})


def test_override_cannot_open_edges_by_mutating_its_input_mask():
    context, bundle = setup_bundle()
    def mutate(func,q,k,v,heads,mask=None,**kw):
        mask.zero_()
        return func(q,k,v,heads,mask=mask,**kw)
    wrapper = RegionalAttentionWrapper(bundle)
    executor = Executor(lambda x,c,o:attention_probe(o,1))
    result = wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context,
                     transformer_options={'optimized_attention_override':mutate})
    assert torch.isneginf(result[0,0,0,8])
    clean = wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context)
    assert torch.isneginf(clean[0,0,0,8])


@pytest.mark.parametrize('rank', [1,3,4])
def test_existing_mask_broadcast_contracts(rank):
    regional = torch.ones(2,1,3,3,dtype=torch.bool)
    if rank == 1:
        existing = torch.tensor([True,False,True])
    elif rank == 3:
        existing = torch.zeros(2,3,3); existing[:,:,1] = -float('inf')
    else:
        existing = torch.zeros(2,2,3,3); existing[:,:,:,1] = -float('inf')
    result = merge_attention_masks(regional,existing,torch.float32,2,2)
    assert torch.isneginf(result[...,1]).all()
    assert (result[...,0] == 0).all()


def test_bias_cast_overflow_is_rejected():
    with pytest.raises(ValueError,match='additive'):
        merge_attention_masks(torch.ones(1,1,2,2,dtype=torch.bool),
                              torch.tensor([[1e20,0.],[0.,0.]]),torch.float16,1,1)


def test_video_reference_and_foreign_positive_are_rejected():
    context,bundle=setup_bundle(); wrapper=RegionalAttentionWrapper(bundle)
    executor=Executor(lambda x,c,o:x)
    with pytest.raises(ValueError,match='video'):
        wrapper(executor,torch.zeros(1,16,2,8,8),torch.ones(1),context)
    with pytest.raises(ValueError,match='reference'):
        wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context,ref_latents=[torch.zeros(1)])
    with pytest.raises(ValueError,match='conditioning'):
        wrapper(executor,torch.zeros(1,16,8,8),torch.ones(1),context[:,:2])
