# SPDX-License-Identifier: CC0-1.0
import inspect
import logging
from functools import wraps

import torch
from comfy import sd, ops, model_management

NODE_CLASS_MAPPINGS = {}

log = logging.getLogger(__name__)

if model_management.is_amd():

    # --- Force PyTorch attention for VAE encode/decode ---
    model_management.pytorch_attention_enabled_vae = lambda: True
    log.info("[ROCm] VAE PyTorch attention forced")

    # --- VAE memory ratio: AMD -> 1.0 ---

    original_init = sd.VAE.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        for name in ("memory_used_encode", "memory_used_decode"):
            fn = getattr(self, name, None)
            if not inspect.isfunction(fn):
                continue

            # 元の関数が実際に使っている補正係数を取得
            ratio = inspect.getclosurevars(fn).nonlocals.get(
                "VAE_KL_MEM_RATIO"
            )

            if not isinstance(ratio, (int, float)) or ratio <= 0:
                continue
            if ratio == 1.0:
                continue

            @wraps(fn)
            def scaled(*a, _fn=fn, _ratio=ratio, **kw):
                return _fn(*a, **kw) / _ratio

            setattr(self, name, scaled)

    sd.VAE.__init__ = patched_init
    log.info("[ROCm] VAE memory ratio workaround installed")


    # --- Conv3d: PR #17166 ---

    conv3d = ops.disable_weight_init.Conv3d
    original_conv = conv3d._conv_forward

    @wraps(original_conv)
    def patched_conv(self, input, weight, bias,
                     autopad=None, *args, **kwargs):

        w = weight
        if autopad == "causal_zero":
            w = weight[:, :, -input.shape[2]:, :, :]

        if (
            input.shape[2:] == w.shape[2:]
            and self.padding == (0, 0, 0)
            and self.dilation == (1, 1, 1)
            and self.groups == 1
        ):
            return torch.nn.functional.linear(
                input.flatten(1), w.flatten(1), bias
            ).reshape(input.shape[0], w.shape[0], 1, 1, 1)

        return original_conv(
            self, input, weight, bias, autopad, *args, **kwargs
        )

    # 本家に同様の最適化が入ったら重複適用しない
    if "linear" not in original_conv.__code__.co_names:
        conv3d._conv_forward = patched_conv
        log.info("[ROCm] Conv3d workaround installed")
    else:
        log.info("[ROCm] Conv3d workaround skipped")
