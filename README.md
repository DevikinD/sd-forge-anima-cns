# sd-forge-anima-cns

Frequency-shaped (**colored**) per-step noise for [sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) NEO — a port of
[ComfyUI-ColoredNoiseDiffusionSampling](https://github.com/akashzeno/ComfyUI-ColoredNoiseDiffusionSampling)
(method: Colored Noise Diffusion Sampling, Davidson/Issachar/Benaim, arXiv:2605.30332), tuned for **Anima**.

Enable the **Anima CNS** accordion, pick a stochastic sampler (Euler a / DPM++ 2M SDE /
DPM++ 2s a RF / ER SDE / …) and shape the spectrum with `alpha_start → alpha_end`.

Quick recipes: **white** `0→0` (baseline) · **CNS-like** `0→-1…-2` · **constant pink/brown** `1→1 … 2→2`.
`energy_scale ≠ 1` is a deliberate "heat" knob.

Console check: `[ColoredNoise] sampling: base=... | colored noise ACTIVE`.
Gamma matrices (`.pt`, `[steps, bins]`) go into `models/colored_noise_gamma/`.
