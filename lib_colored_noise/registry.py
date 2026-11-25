# Port of ComfyUI-ColoredNoiseDiffusionSampling src/sampling/registry.py.
# A sampler is "stochastic" (and thus colorable) iff its k-diffusion function
# exposes a `noise_sampler` parameter — that is the seam where per-step noise
# is injected. Deterministic samplers never call it, so coloring them would be
# a silent no-op and they are excluded.

import functools
import importlib
import inspect

from . import log

# Functions that expose `noise_sampler` but inject ZERO stochastic noise.
# (Forge's sample_res_multistep hardcodes eta=0, so sigma_up == 0.)
_NOOP_NOISE = {"res_multistep", "res_multistep_cfg_pp"}

_cache = None


def _k_diffusion_sampling():
    # `modules/sd_samplers_kdiffusion.py` imports it top-level as `k_diffusion`,
    # so the first name resolves to the exact module object Forge's samplers use.
    for name in ("k_diffusion.sampling", "modules_forge.packages.k_diffusion.sampling"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def stochastic_samplers():
    """Return `{base_name: sample_fn}` for every colorable stochastic sampler, sorted by name.

    The `sigmas` requirement excludes restart/unipc/ddim-style functions with a
    different signature; `_gpu` variants are deduped (the noise device is taken
    from the latent `x`, so the split is irrelevant here). Note: Forge's CFG++
    variants (e.g. euler_ancestral_cfg_pp) DO take a noise_sampler, unlike the
    ComfyUI originals — introspection picks them up automatically.
    """
    global _cache
    if _cache is None:
        kds = _k_diffusion_sampling()
        out = {}
        if kds is None:
            log.warning("could not import k_diffusion; the Base Sampler list is limited to Auto")
        else:
            for name in dir(kds):
                if not name.startswith("sample_"):
                    continue
                fn = getattr(kds, name)
                if not callable(fn):
                    continue
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    continue
                if "noise_sampler" not in params or "sigmas" not in params:
                    continue
                base = name[len("sample_"):]
                if base.endswith("_gpu") or base in _NOOP_NOISE:
                    continue
                out[base] = fn
        _cache = dict(sorted(out.items()))
    return _cache


@functools.lru_cache(maxsize=None)
def accepted_params(fn):
    try:
        return frozenset(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        return frozenset()


def base_name_of(fn):
    name = getattr(fn, "__name__", "") or "sampler"
    return name[len("sample_"):] if name.startswith("sample_") else name
