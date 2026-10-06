"""Per-model Krea2 regional attention (prompt-only Gate A).

Adapted from januspluto/Krea2-Regional, MIT; see THIRD_PARTY_NOTICES.md.
No global model methods or weights are replaced by this implementation.
"""
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from types import CodeType
import uuid

import torch

from .diagnostics import emit_debug
from .regional_attention import build_attention_masks, build_region_owners


CURRENT_REGIONAL_CALL = ContextVar('krea2_slider_regional_call', default=None)
WRAPPER_KEY = 'krea2_slider_regional_attention'


def _revision(directory):
    try:
        return subprocess.check_output(['git', '-c', f'safe.directory={directory.as_posix()}',
                                        '-C', str(directory), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _run_provenance():
    """Optional diagnostic metadata must not prevent sampling from starting."""
    extension = sys.modules.get('comfy.patcher_extension')
    sources = {
        'comfy_commit': lambda: _revision(Path(extension.__file__).resolve().parents[1]),
        'extension_commit': lambda: _revision(Path(__file__).resolve().parents[1]),
        'runtime_sha256': lambda: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result, errors = {}, []
    for name, get_value in sources.items():
        try:
            result[name] = get_value()
        except (OSError, TypeError, ValueError, AttributeError):
            result[name] = 'unknown'
            errors.append(name)
    if errors:
        result['metadata_unavailable'] = errors
    return result


def _supported_native_replacement(override):
    """Recognize the core setter closure and only its mask-aware native backends."""
    patcher_module = sys.modules.get('comfy.model_patcher')
    attention_module = sys.modules.get('comfy.ldm.modules.attention')
    setter = getattr(getattr(patcher_module, 'ModelPatcher', None), 'set_model_optimized_attention', None)
    setter_code = getattr(setter, '__code__', None)
    override_code = getattr(override, '__code__', None)
    if setter_code is None or not any(override_code is code for code in setter_code.co_consts
                                      if isinstance(code, CodeType)):
        return False
    closure = dict(zip(override_code.co_freevars, override.__closure__ or ()))
    cell = closure.get('optimized_attention')
    if cell is None:
        return False
    supported = ('attention_basic', 'attention_pytorch', 'attention_split', 'attention_sub_quad',
                 'attention_xformers', 'attention_sage', 'attention_flash', 'attention_comfy_kitchen_int8')
    return any(cell.cell_contents is getattr(attention_module, name, None) for name in supported)


@dataclass(frozen=True)
class RegionSpec:
    conditioning: object
    mask: torch.Tensor
    loras: tuple = ()


@dataclass(frozen=True)
class RegionalBundle:
    segments: tuple
    masks: tuple
    isolation: str = 'strict'
    debug_logging: bool = False


def _conditioning_tensor(conditioning, feature_dim):
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        raise ValueError('Regional conditioning must contain exactly one entry')
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
        raise ValueError('Invalid regional conditioning entry')
    tensor, metadata = entry
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError('Regional conditioning requires shape (1,tokens,features)')
    if tensor.shape[-1] != feature_dim or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f'Regional conditioning requires {feature_dim} finite features')
    unsupported = set(metadata) - {'pooled_output', 'attention_mask'}
    if unsupported:
        raise ValueError(f'Regional conditioning has unsupported metadata: {sorted(unsupported)}; use plain CLIP Text Encode')
    mask = metadata.get('attention_mask')
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.numel() != tensor.shape[1]:
            raise ValueError('Conditioning attention_mask must describe one text sequence')
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('Conditioning attention_mask must be binary padding metadata')
        tensor = tensor[:, mask.reshape(-1).to(device=tensor.device, dtype=torch.bool)]
    if tensor.shape[1] == 0:
        raise ValueError('Regional conditioning cannot be empty')
    return tensor, {k: v for k, v in metadata.items() if k != 'attention_mask'}


def compose_regional_conditioning(base, background, regions, *, feature_dim=30720,
                                  max_tokens=512, isolation='strict', debug_logging=False):
    if isolation not in ('strict', '_unrestricted'):
        raise ValueError('Only strict isolation is available before Gate A')
    if not 1 <= len(regions) <= 4 or not all(isinstance(r, RegionSpec) for r in regions):
        raise ValueError('Connect a chain of 1–4 Krea2 Regional Prompt Regions')
    if any(r.loras for r in regions):
        raise ValueError('Regional LoRA requires Gate A approval; this build is prompt-only')
    masks = tuple(r.mask.detach().to(device='cpu', dtype=torch.float32, copy=True)
                  if isinstance(r.mask, torch.Tensor) else r.mask for r in regions)
    # Validate canvas/value contracts now; token-size validation happens at forward.
    canvas = masks[0].shape[-2:] if isinstance(masks[0], torch.Tensor) else (1, 1)
    build_region_owners(masks, canvas)
    tensors, segments, offset = [], [], 0
    base_metadata = None
    for condition in [base, *(r.conditioning for r in regions), background]:
        tensor, metadata = _conditioning_tensor(condition, feature_dim)
        if base_metadata is None:
            base_metadata = metadata
        segments.append((offset, offset + tensor.shape[1]))
        offset += tensor.shape[1]
        tensors.append(tensor)
    if offset > max_tokens:
        raise ValueError(f'Combined regional text has {offset} tokens; limit is {max_tokens}; shorten prompts')
    if any(t.device != tensors[0].device or t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError('Regional conditioning must have the same device and dtype')
    bundle = RegionalBundle(tuple(segments), masks, isolation, debug_logging)
    return [[torch.cat(tensors, dim=1), base_metadata]], bundle


def merge_attention_masks(regional, existing, dtype, batch, heads):
    """Intersect restrictions and preserve additive biases; True means allowed."""
    result = torch.zeros(regional.shape, device=regional.device, dtype=dtype)
    result.masked_fill_(~regional, -float('inf'))
    if existing is not None:
        if not isinstance(existing, torch.Tensor):
            raise ValueError('Unsupported existing attention mask')
        previous = existing.to(device=regional.device)
        q, k = regional.shape[-2:]
        if previous.ndim == 1:
            previous = previous[None, None, None, :]
        elif previous.ndim == 2:
            if previous.shape == (q, k):
                previous = previous[None, None]
            elif previous.shape == (batch, k):
                previous = previous[:, None, None]
            else:
                raise ValueError('Unknown two-dimensional attention mask shape')
        elif previous.ndim == 3:
            previous = previous[:, None]
        elif previous.ndim != 4:
            raise ValueError('Unknown attention mask rank')
        if any(actual not in (1, expected) for actual, expected in zip(previous.shape, (batch, heads, q, k))):
            raise ValueError('Existing attention mask cannot broadcast to the regional attention')
        if previous.dtype == torch.bool:
            previous = torch.zeros_like(previous, dtype=dtype).masked_fill_(~previous, -float('inf'))
        elif not previous.is_floating_point():
            raise ValueError('Existing attention mask must be boolean or additive floating point')
        elif torch.isnan(previous).any() or torch.isposinf(previous).any():
            raise ValueError('Existing attention mask has invalid additive values')
        previous = previous.to(dtype=dtype)
        if torch.isnan(previous).any() or torch.isposinf(previous).any():
            raise ValueError('Existing additive attention mask overflows the attention dtype')
        result = result + previous
    if not torch.isfinite(result).any(dim=-1).all():
        raise ValueError('Regional attention leaves a query without any valid key')
    return result


class RegionalAttentionWrapper:
    def __init__(self, bundle):
        self.bundle = bundle
        self.reset()

    def reset(self):
        self.run_id = uuid.uuid4().hex[:12]
        self.forward_count = 0
        self.cache = {}

    def _log(self, event, **values):
        if self.bundle.debug_logging:
            emit_debug({'event': event, 'run_id': self.run_id, **values})

    def __call__(self, executor, x, timesteps, context, *args, **kwargs):
        # Native Krea2 _forward: attention_mask, ref_latents, transformer_options.
        positional = list(args)
        if len(positional) > 3:
            raise ValueError('Unsupported Krea2 forward arguments')
        supplied = kwargs.get('transformer_options', positional[2] if len(positional) > 2 else {})
        options = dict(supplied or {})
        flags = options.get('cond_or_uncond', [0])
        if not flags or any(flag not in (0, 1) for flag in flags) or x.shape[0] % len(flags):
            raise ValueError('Unsupported cond_or_uncond batch layout')
        if all(flag == 1 for flag in flags):
            return executor(x, timesteps, context, *args, **kwargs)
        patches = options.get('patches', {})
        if any(patches.get(name) for name in ('post_input', 'attn1_patch', 'attn1_output_patch')):
            raise ValueError('Regional attention cannot verify custom token/attention patches; remove them')
        if x.ndim not in (4, 5) or (x.ndim == 5 and x.shape[2] != 1):
            raise ValueError('Regional attention supports static images only; video is unsupported')
        references = kwargs.get('ref_latents', positional[1] if len(positional) > 1 else None)
        if references is not None and len(references) != 0:
            raise ValueError('Regional attention does not support reference latents')
        text_length = self.bundle.segments[-1][1]
        if context.ndim != 3 or context.shape[:2] != (x.shape[0], text_length):
            raise ValueError('Positive conditioning differs from the composed regional conditioning')
        model = executor.class_obj
        patch = getattr(model, 'patch', None)
        layers = getattr(model, 'txtlayers', None)
        if not isinstance(patch, int) or patch < 1 or not isinstance(layers, int):
            raise ValueError('Regional attention requires a native Krea2 SingleStreamDiT')
        token_hw = tuple((size + patch - 1) // patch for size in x.shape[-2:])
        positive = tuple(flag == 0 for flag in flags for _ in range(x.shape[0] // len(flags)))
        key = (token_hw, positive, x.device, x.dtype)
        if key not in self.cache:
            owners = build_region_owners(self.bundle.masks, token_hw)
            joint, text = build_attention_masks(self.bundle.segments, owners, self.bundle.isolation)
            def batch_mask(mask):
                result = mask[None, None].expand(x.shape[0], 1, *mask.shape).clone()
                result[~torch.tensor(positive)] = True
                return result.to(device=x.device)
            self.cache[key] = (batch_mask(joint), batch_mask(text), {})
            counts = torch.bincount(owners, minlength=len(self.bundle.masks) + 1).tolist()
            bounds = []
            for region in range(len(counts)):
                ys, xs = torch.where(owners.reshape(token_hw) == region)
                bounds.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                              if len(xs) else None)
            self._log('regional_geometry', latent_shape=list(x.shape), token_hw=token_hw,
                      segments=self.bundle.segments, owner_counts=counts, owner_bounds_xyxy=bounds,
                      background_owner=len(self.bundle.masks), isolation=self.bundle.isolation,
                      mask_bytes=joint.numel() + text.numel(),
                      mask_hashes=[hashlib.sha256(m.numpy().tobytes()).hexdigest() for m in self.bundle.masks],
                      regional_lora_count=0, forbidden_edges=int((~joint).sum()),
                      base_to_image_allowed_edges=int(joint[:self.bundle.segments[0][1], text_length:].sum()))
        joint_mask, text_mask, additive_cache = self.cache[key]
        previous_override = options.get('optimized_attention_override')
        native_replacement = _supported_native_replacement(previous_override)
        counts = {'joint': 0, 'text': 0, 'layerwise': 0}

        def override(func, q, k, v, heads, mask=None, **attention_kwargs):
            if not isinstance(q, torch.Tensor) or q.ndim not in (3, 4):
                raise ValueError('Unknown native attention tensor format')
            batch, length = q.shape[0], q.shape[-2]
            if k.shape[-2] != length:
                raise ValueError('Regional attention requires native self-attention')
            if length == layers and batch == x.shape[0] * text_length:
                # txtfusion across encoder taps, independent for each text token.
                counts['layerwise'] += 1
                if previous_override:
                    return previous_override(func, q, k, v, heads, mask=mask, **attention_kwargs)
                return func(q, k, v, heads, mask=mask, **attention_kwargs)
            if batch != x.shape[0]:
                raise ValueError('Unknown attention batch layout; regional mask was not applied')
            if length == joint_mask.shape[-1]:
                regional, kind = joint_mask, 'joint'
            elif length == text_length:
                regional, kind = text_mask, 'text'
            else:
                raise ValueError(f'Unknown attention length {length}; regional mask was not applied')
            counts[kind] += 1
            cache_key = (kind, q.dtype)
            if cache_key not in additive_cache:
                additive_cache[cache_key] = merge_attention_masks(regional, None, q.dtype, batch, heads)
            merged = (additive_cache[cache_key] if mask is None else
                      merge_attention_masks(regional, mask, q.dtype, batch, heads))
            if native_replacement:
                # The core setter deliberately replaces func. Its known native backend
                # accepts additive masks; unknown replacements still use the guard.
                return previous_override(func, q, k, v, heads, mask=merged, **attention_kwargs)
            called = False
            def guarded(q1, k1, v1, heads1, mask=None, **inner_kwargs):
                nonlocal called
                called = True
                # The override already receives merged biases. Do not add those twice.
                if q1.shape != q.shape or k1.shape != k.shape or heads1 != heads:
                    raise ValueError('An attention override changed the regional token layout')
                if mask is None or mask is merged:
                    final = merged
                else:
                    final = merge_attention_masks(regional, mask, q1.dtype, batch, heads1)
                    final = final.masked_fill(torch.isneginf(merged), -float('inf'))
                return func(q1, k1, v1, heads1, mask=final, **inner_kwargs)
            if previous_override:
                # Unknown adapters may edit their argument in place. Protect the run cache
                # and keep an immutable restriction mask for the guarded backend.
                result = previous_override(guarded, q, k, v, heads, mask=merged.clone(), **attention_kwargs)
                if not called:
                    raise RuntimeError('An attention override bypassed the guarded regional backend')
                return result
            return func(q, k, v, heads, mask=merged, **attention_kwargs)

        options['optimized_attention_override'] = override
        if len(positional) > 2:
            positional[2] = options
        else:
            kwargs['transformer_options'] = options
        self.forward_count += 1
        token = CURRENT_REGIONAL_CALL.set({'wrapper': self, 'positive_rows': positive})
        try:
            result = executor(x, timesteps, context, *positional, **kwargs)
            if counts['joint'] == 0:
                raise RuntimeError('Regional attention did not reach a joint attention backend')
            if counts['text'] == 0:
                raise RuntimeError('Regional attention did not reach a text-fusion attention backend')
            self._log('regional_forward', forward_count=self.forward_count,
                      cond_or_uncond=flags, attention_calls=counts,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(x.device) if x.is_cuda else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(x.device) if x.is_cuda else 0)
            return result
        except Exception as error:
            self.cache.clear()
            self._log('regional_error', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            CURRENT_REGIONAL_CALL.reset(token)


def apply_regional_attention(model, base, background, regions, isolation='strict', debug_logging=False):
    from comfy.patcher_extension import CallbacksMP, WrappersMP

    config = getattr(getattr(getattr(model, 'model', None), 'model_config', None), 'unet_config', {})
    if config.get('image_model') != 'krea2':
        raise ValueError('Connect a native Krea2 diffusion MODEL')
    if any(getattr(model, name, None) for name in ('forced_hooks', 'hook_patches', 'current_hooks')):
        raise ValueError('Remove native LoRA Hooks from the Regional Attention model path')
    if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY):
        raise ValueError('Regional Attention is already applied; connect the preceding model')
    for wrapper in model.get_all_wrappers(WrappersMP.DIFFUSION_MODEL):
        if 'krea2_regional' in getattr(wrapper, '__module__', ''):
            raise ValueError('Do not combine Krea2 Regional Attention engines on one model path')
    diffusion = getattr(model.model, 'diffusion_model', None)
    feature_dim = getattr(diffusion, 'txtlayers', 12) * getattr(diffusion, 'txtdim', 2560)
    conditioning, bundle = compose_regional_conditioning(base, background, regions, feature_dim=feature_dim,
                                                       isolation=isolation, debug_logging=debug_logging)
    clone = model.clone()
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, RegionalAttentionWrapper(bundle))

    def on_clone(original, copied):
        copied.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        copied.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, RegionalAttentionWrapper(bundle))

    def on_pre_run(patcher):
        for wrapper in patcher.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY):
            wrapper.reset()
            if bundle.debug_logging:
                wrapper._log('regional_run_start', torch_version=torch.__version__, regional_lora_count=0,
                             **_run_provenance(),
                             peak_scope='process CUDA peak; not reset by this node')

    clone.add_callback_with_key(CallbacksMP.ON_CLONE, WRAPPER_KEY, on_clone)
    clone.add_callback_with_key(CallbacksMP.ON_PRE_RUN, WRAPPER_KEY, on_pre_run)
    return clone, conditioning
