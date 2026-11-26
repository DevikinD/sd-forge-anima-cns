# Port of ComfyUI-ColoredNoiseDiffusionSampling src/engine/colored_noise.py,
# bridged onto Forge's sampler seam + p.modified_noise initial-noise seam.

from dataclasses import dataclass
from typing import Optional

import torch

from . import log
from .schedule import gamma_row_at, gamma_scaling, interp_alpha, progress_from_sigma, load_gamma_matrix
from .spectral import color_tensor, parametric_scaling

PARAMETRIC_BINS = 256


@dataclass
class ColorParams:
    mode: str = "parametric"                  # "parametric" | "gamma_matrix"
    alpha_start: float = 0.0
    alpha_end: float = -1.0
    interpolation: str = "linear"             # "linear" | "exponential"
    exp_sharpness: float = 4.0
    gamma_matrix: Optional[torch.Tensor] = None
    gamma_divider: float = 1.0
    gamma_shaping: str = "none"               # "none" | "sqrt" | "power"
    power_gamma: float = 1.0
    alpha_tilting: float = 0.0
    energy_scale: float = 1.0


def scaling_for_progress(params: ColorParams, progress, device=torch.device("cpu")):
    """Per-bin amplitude vector for this trajectory progress (mode-dispatched)."""
    if params.mode == "gamma_matrix" and params.gamma_matrix is not None:
        row = gamma_row_at(params.gamma_matrix.to(device), progress)
        return gamma_scaling(row, params.gamma_divider, params.gamma_shaping, params.power_gamma, params.alpha_tilting)
    alpha = interp_alpha(params.alpha_start, params.alpha_end, progress, params.interpolation, params.exp_sharpness)
    return parametric_scaling(PARAMETRIC_BINS, alpha, device)


def _make_generator(x, seed):
    if seed is None:
        return None
    # mirror comfy.k_diffusion default_noise_sampler's device/seed conventions
    s = int(seed) + (1 if x.device == torch.device("cpu") else 0)
    g = torch.Generator(device=x.device)
    g.manual_seed(s)
    return g


def make_colored_noise_sampler(x, sigmas, params: ColorParams, seed):
    """Build a `noise_sampler(sigma, sigma_next) -> unit-variance colored noise`, closed over x.

    Built inside the wrapped sampler function (where the real `x` and final `sigmas`
    are known), then passed into the k-diffusion base sampler — the exact seam the
    ComfyUI nodes use, and the same one Forge's BrownianTree samplers plug into.
    """
    sigma_max = float(sigmas[0])
    positives = [float(s) for s in sigmas if float(s) > 0]
    sigma_min = min(positives) if positives else 1e-3
    generator = _make_generator(x, seed)

    def noise_sampler(sigma, sigma_next):
        white = torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)
        progress = progress_from_sigma(sigma, sigma_max, sigma_min)
        scaling = scaling_for_progress(params, progress, device=torch.device("cpu"))
        if log.debug_enabled():
            mid = scaling.shape[0] // 2
            log.debug("step sigma=%.4f progress=%.3f | scale low/mid/high=%.3f/%.3f/%.3f",
                      float(sigma), progress, float(scaling[0]), float(scaling[mid]), float(scaling[-1]))
        return color_tensor(white, scaling, energy_scale=params.energy_scale)

    return noise_sampler


def color_initial_noise(noise: torch.Tensor, params: ColorParams) -> torch.Tensor:
    """Spectrally shape Forge's initial noise (the upstream NOISE node's job).

    Colored per batch item (matching the upstream NOISE node's per-item unit
    variance), then the incoming RMS is restored so Forge-side scaling of the
    noise (e.g. img2img's `initial_noise_multiplier`, applied before this hook)
    is preserved. Works on [B,C,H,W] and Anima's [B,C,T=1,H,W] latents.
    """
    scaling = scaling_for_progress(params, 0.0, device=torch.device("cpu"))
    incoming_std = float(noise.float().std())
    items = [color_tensor(noise[i : i + 1].float(), scaling, energy_scale=params.energy_scale) for i in range(noise.shape[0])]
    shaped = torch.cat(items, dim=0)
    if incoming_std > 1e-9:
        shaped = shaped * incoming_std
    return shaped.to(noise.dtype)
