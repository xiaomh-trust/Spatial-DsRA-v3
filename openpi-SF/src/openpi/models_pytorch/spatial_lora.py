"""Training-only Q/V LoRA for the existing PyTorch PaliGemma implementation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpatialLoRALinear(nn.Linear):
    # [Spatial-DsRA v3]: Preserve base weight keys and attention interfaces, add rank-32 Q/V updates.
    def __init__(self, base, rank):
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = min(rank, 16) / rank  # Match the existing OpenVLA LoRA scaling convention.

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (update * self.scaling).to(base.dtype)


def configure_spatial_lora(model, rank=32):
    # [Spatial-DsRA v3]: Freeze VLM language weights; preserve vision encoder base training setting.
    language = model.paligemma_with_expert.paligemma.language_model
    language.requires_grad_(False)
    for block in language.layers:
        for name in ("q_proj", "v_proj"):
            setattr(block.self_attn, name, SpatialLoRALinear(getattr(block.self_attn, name), rank))
    # Action expert and action projections remain fully trainable as in the base pi0 trainer.


def deployment_state_dict(model):
    # [Spatial-DsRA v3]: Export plain pi0 weights with merged LoRA, no auxiliary modules.
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
             if not k.endswith(("lora_A", "lora_B"))}
    for name, module in model.named_modules():
        if isinstance(module, SpatialLoRALinear):
            delta = (module.lora_B.detach().float() @ module.lora_A.detach().float()) * module.scaling
            key = name + ".weight"
            state[key] = (module.weight.detach().float() + delta).to(module.weight.dtype).cpu().contiguous()
    return state
