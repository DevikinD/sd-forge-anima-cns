# Gamma matrices

Copy the two bundled matrices from the ComfyUI pack
(`ComfyUI-ColoredNoiseDiffusionSampling/gamma_matrices/`) into either:

- `models/colored_noise_gamma/`  (user folder; survives extension updates), or
- `sd-forge-anima-cns/gamma_matrices/` (this folder)

- `gamma_matrix_scaled.pt`
- `gamma_matrix_scaled_cfg_1.5.pt`

Both are raw `[steps, bins]` float32 tensors (`torch.save`). Drop your own
matrices into `models/colored_noise_gamma/` and use "Reload UI" to pick them up.

Provenance: Hadar Davidson, Roy Issachar & Sagie Benaim,
https://github.com/hadardavidson/colored-noise-sampling (MIT), arXiv:2605.30332.

Note: on Anima (or any non-SiT model) these matrices are a documented
heuristic, not a calibrated schedule — prefer `parametric` mode.
