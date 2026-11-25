import functools
import inspect
import os

import gradio as gr
import torch

from modules import scripts
from modules.infotext_utils import PasteField
from modules.paths_internal import models_path
from modules.ui_components import InputAccordion

from lib_colored_noise import engine, log, registry
from lib_colored_noise.engine import ColorParams


AUTO = "Auto (use current sampler)"
NONE = "none (parametric)"
GAMMA_FOLDER = "colored_noise_gamma"
_GAMMA_EXTS = (".pt", ".pth")
_BUNDLED_GAMMA = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "gamma_matrices")


def _gamma_dirs():
    return [os.path.join(models_path, GAMMA_FOLDER), _BUNDLED_GAMMA]


def gamma_choices():
    seen, files = set(), []
    for directory in _gamma_dirs():
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(_GAMMA_EXTS) and name not in seen:
                seen.add(name)
                files.append(name)
    return [NONE] + files


def gamma_path(name):
    for directory in _gamma_dirs():
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


class ColoredNoiseForForge(scripts.Script):
    """Frequency-shaped (colored) stochastic noise — Forge port of
    ComfyUI-ColoredNoiseDiffusionSampling, tuned for the Anima model.

    - Per-step noise: re-points the k-diffusion sampler's `noise_sampler` seam
      (the same seam the upstream nodes plug into), overriding both the default
      white noise and Forge's BrownianTree noise sampler.
    - Initial noise: optionally reshapes Forge's initial latent noise via
      `p.modified_noise` (applied to txt2img, img2img and the Hires pass).
    """

    sorting_priority = 2027

    def title(self):
        return "Anima CNS"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # region UI

    def ui(self, is_img2img):
        samplers = list(registry.stochastic_samplers().keys())

        with InputAccordion(False, label=self.title()) as enable:
            base_sampler = gr.Dropdown(
                value=AUTO,
                choices=[AUTO] + samplers,
                label="Base Sampler",
                info="Stochastic sampler whose per-step noise is colored; Auto = whatever the Sampling Method dropdown selected",
            )
            with gr.Row():
                mode = gr.Radio(choices=["parametric", "gamma_matrix"], value="parametric", label="Mode")
            with gr.Row():
                alpha_start = gr.Slider(minimum=-8.0, maximum=8.0, step=0.05, value=0.0, label="Alpha start", info="Spectral exponent at the start (high sigma); 0 = white, + = pink/brown, - = blue/violet")
                alpha_end = gr.Slider(minimum=-8.0, maximum=8.0, step=0.05, value=-1.0, label="Alpha end", info="Spectral exponent at the end (low sigma); interpolated across the trajectory")
            with gr.Row():
                interpolation = gr.Radio(choices=["linear", "exponential"], value="linear", label="Interpolation")
                exp_sharpness = gr.Slider(minimum=0.1, maximum=16.0, step=0.1, value=4.0, label="Exp. sharpness", info="Only used by exponential interpolation")
            with gr.Row():
                energy_scale = gr.Slider(minimum=0.0, maximum=4.0, step=0.01, value=1.0, label="Energy scale", info="Scales the noise std after unit-variance renorm; 1.0 = neutral")

            with gr.Group():
                with gr.Row():
                    gamma_matrix = gr.Dropdown(choices=gamma_choices(), value=NONE, label="Gamma matrix", info="gamma_matrix mode only; put [steps, bins] .pt tensors into models/colored_noise_gamma")
                    gamma_divider = gr.Slider(minimum=0.1, maximum=10.0, step=0.01, value=1.0, label="Gamma divider")
                with gr.Row():
                    gamma_shaping = gr.Radio(choices=["none", "sqrt", "power"], value="none", label="Gamma shaping")
                    power_gamma = gr.Slider(minimum=0.1, maximum=8.0, step=0.05, value=1.0, label="Power gamma")
                    alpha_tilting = gr.Slider(minimum=-8.0, maximum=8.0, step=0.05, value=0.0, label="Alpha tilting")

            color_initial_noise = gr.Checkbox(value=False, label="Color initial noise", info="Also frequency-shape the initial latent noise (constant color = Alpha start)")

        self.infotext_fields = [
            PasteField(enable, "Colored Noise", api="colored_noise"),
            PasteField(base_sampler, "CN Base Sampler", api="colored_noise_base"),
            PasteField(mode, "CN Mode", api="colored_noise_mode"),
            PasteField(alpha_start, "CN Alpha start", api="colored_noise_alpha_start"),
            PasteField(alpha_end, "CN Alpha end", api="colored_noise_alpha_end"),
            PasteField(interpolation, "CN Interpolation", api="colored_noise_interpolation"),
            PasteField(exp_sharpness, "CN Exp sharpness", api="colored_noise_exp_sharpness"),
            PasteField(energy_scale, "CN Energy scale", api="colored_noise_energy_scale"),
            PasteField(gamma_matrix, "CN Gamma matrix", api="colored_noise_gamma_matrix"),
            PasteField(gamma_divider, "CN Gamma divider", api="colored_noise_gamma_divider"),
            PasteField(gamma_shaping, "CN Gamma shaping", api="colored_noise_gamma_shaping"),
            PasteField(power_gamma, "CN Power gamma", api="colored_noise_power_gamma"),
            PasteField(alpha_tilting, "CN Alpha tilting", api="colored_noise_alpha_tilting"),
            PasteField(color_initial_noise, "CN Color initial noise", api="colored_noise_initial"),
        ]
        self.paste_field_names = [field.label for field in self.infotext_fields]

        return [enable, base_sampler, mode, alpha_start, alpha_end, interpolation, exp_sharpness, energy_scale, gamma_matrix, gamma_divider, gamma_shaping, power_gamma, alpha_tilting, color_initial_noise]

    # endregion

    # region Processing

    def process_before_every_sampling(self, p, enable: bool, base_sampler: str, mode: str, alpha_start: float, alpha_end: float, interpolation: str, exp_sharpness: float, energy_scale: float, gamma_matrix: str, gamma_divider: float, gamma_shaping: str, power_gamma: float, alpha_tilting: float, color_initial_noise: bool, *args, **kwargs):
        if not enable:
            return

        from modules.sd_samplers_kdiffusion import KDiffusionSampler

        sampler = getattr(p, "sampler", None)
        if not isinstance(sampler, KDiffusionSampler):
            log.warn_once("non_kdiffusion", "Colored Noise: '{}' is not a k-diffusion sampler; skipped".format(getattr(p, "sampler_name", "?")))
            return

        self._warn_model_scope()

        params = self._resolve_params(mode, alpha_start, alpha_end, interpolation, exp_sharpness, energy_scale, gamma_matrix, gamma_divider, gamma_shaping, power_gamma, alpha_tilting)

        # 1) colored initial noise — replaces the noise Forge is about to use,
        #    before the sampler multiplies it by sigma (mirrors the upstream NOISE node)
        noise = kwargs.get("noise", None)
        if color_initial_noise and isinstance(noise, torch.Tensor):
            p.modified_noise = engine.color_initial_noise(noise, ColorParams(mode="parametric", alpha_start=alpha_start, alpha_end=alpha_start, energy_scale=energy_scale))
            log.info("colored initial noise: alpha=%.2f | energy=%.2f | seed=%s", alpha_start, energy_scale, (p.seeds or [-1])[0])

        # 2) colored per-step noise — mirror of the upstream SamplerSelect node.
        #    Note: img2img reuses one sampler object across iterations, so always
        #    wrap the pristine original instead of the (possibly already wrapped) func.
        original_func = getattr(sampler, "_colored_noise_original", None)
        if original_func is None:
            original_func = sampler.func
        sampler._colored_noise_original = original_func

        if base_sampler == AUTO:
            target = original_func
            if "noise_sampler" not in registry.accepted_params(target):
                log.warn_once("not_stochastic", "Colored Noise: '{}' injects no per-step noise (deterministic or non-stochastic base); pick a stochastic sampler such as Euler a / DPM++ 2M SDE / DPM++ 2s a RF".format(p.sampler_name))
                return
        else:
            target = registry.stochastic_samplers().get(base_sampler)
            if target is None:
                log.warn_once("bad_base", "Colored Noise: base sampler '{}' unavailable; skipped".format(base_sampler))
                return

        base_label = registry.base_name_of(target)
        accepted = registry.accepted_params(target)
        seed = int(p.seeds[0]) if getattr(p, "seeds", None) else None

        @functools.wraps(original_func)
        def colored_func(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            # `functools.wraps` keeps inspect.signature(self.func) == the UI sampler's
            # signature, so Forge's initialize() still forwards eta/s_noise/etc. here.
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            if "eta" in accepted and float(kwargs.get("eta", 1.0)) == 0.0:
                log.warn_once("eta0_" + base_label, "Colored Noise: eta=0 on '{}' makes each step deterministic; colored noise has no effect (raise 'Eta for k-diffusion samplers' in Settings)".format(base_label))
            # overrides both default_noise_sampler and Forge's BrownianTree sampler
            kwargs["noise_sampler"] = engine.make_colored_noise_sampler(x, sigmas, params, seed)
            return target(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs)

        sampler.func = colored_func

        if params.mode == "gamma_matrix":
            log.info("sampling: base=%s | gamma_matrix (divider=%.2f, shaping=%s) | energy=%.2f | %d steps | colored noise ACTIVE", base_label, params.gamma_divider, params.gamma_shaping, params.energy_scale, p.steps)
        else:
            log.info("sampling: base=%s | parametric alpha %.2f->%.2f (%s) | energy=%.2f | %d steps | colored noise ACTIVE", base_label, params.alpha_start, params.alpha_end, params.interpolation, params.energy_scale, p.steps)

        extra = {
            "Colored Noise": True,
            "CN Base Sampler": base_sampler,
            "CN Mode": params.mode,
            "CN Alpha start": params.alpha_start,
            "CN Alpha end": params.alpha_end,
            "CN Interpolation": params.interpolation,
            "CN Exp sharpness": params.exp_sharpness,
            "CN Energy scale": params.energy_scale,
        }
        if params.mode == "gamma_matrix":
            extra.update({
                "CN Gamma matrix": gamma_matrix,
                "CN Gamma divider": params.gamma_divider,
                "CN Gamma shaping": params.gamma_shaping,
                "CN Power gamma": params.power_gamma,
                "CN Alpha tilting": params.alpha_tilting,
            })
        if color_initial_noise:
            extra["CN Color initial noise"] = True
        p.extra_generation_params.update(extra)

    @staticmethod
    def _warn_model_scope():
        try:
            from backend.args import dynamic_args
            if not dynamic_args.anima:
                log.warn_once("not_anima", "Colored Noise: this port targets the Anima model; the current checkpoint is not Anima and is untested")
        except Exception:
            pass

    @staticmethod
    def _resolve_params(mode, alpha_start, alpha_end, interpolation, exp_sharpness, energy_scale, gamma_matrix, gamma_divider, gamma_shaping, power_gamma, alpha_tilting):
        gm = None
        effective = "parametric"
        if mode == "gamma_matrix" and gamma_matrix != NONE:
            path = gamma_path(gamma_matrix)
            if path is None:
                log.warn_once("gamma_missing", "Colored Noise: gamma matrix '{}' not found; falling back to parametric mode".format(gamma_matrix))
            else:
                gm = engine.load_gamma_matrix(path)
                effective = "gamma_matrix"
        return ColorParams(
            mode=effective,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            interpolation=interpolation,
            exp_sharpness=exp_sharpness,
            gamma_matrix=gm,
            gamma_divider=gamma_divider,
            gamma_shaping=gamma_shaping,
            power_gamma=power_gamma,
            alpha_tilting=alpha_tilting,
            energy_scale=energy_scale,
        )

    # endregion
