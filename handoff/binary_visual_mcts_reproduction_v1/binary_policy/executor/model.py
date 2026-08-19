"""Non-invasive wrapper around a frozen Qwen2.5-VL conditional LM."""

from __future__ import annotations

from torch import nn

from .inputs import resolve_causal_lm, resolve_decoder


class BinaryQwen25VL(nn.Module):
    _is_binary_qwen_wrapper = True

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return cls(Qwen2_5_VLForConditionalGeneration.from_pretrained(*args, **kwargs))

    @property
    def config(self):
        return self.base_model.config

    @property
    def decoder(self):
        return resolve_decoder(self.base_model)

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self
