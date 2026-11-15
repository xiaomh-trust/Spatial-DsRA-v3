"""Spatial-DsRA v3: training-only full-layer spatial distillation."""
import math
import hashlib
import os
import uuid
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class SpatialDsRA(nn.Module):
    # [Spatial-DsRA v3]: Shared dv->db->dg mapping, shared M, independent rho[L,r].
    def __init__(self, llm_dim, teacher_dim, num_layers, bottleneck_dim=256, rank=32,
                 a_max=0.5, warmup_steps=20000, ramp_steps=40000,
                 freeze_step=None, beta_r=None, eta=None):
        super().__init__()
        if not 0 < rank <= bottleneck_dim or not 0 < a_max <= 1 or num_layers < 1:
            raise ValueError("Invalid Spatial-DsRA dimensions or modulation bound")
        if freeze_step is None or beta_r is None or eta is None:
            raise ValueError("Specify freeze_step, beta_r and eta: the v3 proposal does not assign values")
        if warmup_steps < 0 or ramp_steps <= 0 or freeze_step < warmup_steps + ramp_steps:
            raise ValueError("Stage III must start after the modulation ramp")
        if beta_r < 0 or eta < 0:
            raise ValueError("Regularization weights must be nonnegative")
        self.llm_dim, self.teacher_dim, self.num_layers = llm_dim, teacher_dim, num_layers
        self.norm = nn.LayerNorm(llm_dim)
        self.down = nn.Linear(llm_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, teacher_dim)
        self.M = nn.Parameter(torch.empty(bottleneck_dim, rank))
        nn.init.orthogonal_(self.M)
        self.rho = nn.Parameter(torch.zeros(num_layers, rank))
        self.a_max, self.beta_r, self.eta = a_max, beta_r, eta
        self.warmup_steps, self.ramp_steps, self.freeze_step = warmup_steps, ramp_steps, freeze_step
        self.stage, self.gamma = 1, 0.0

    def set_step(self, step):
        # [Spatial-DsRA v3]: Register all parameters with DDP/optimizer first; freeze by stage afterwards.
        self.stage = 1 if step < self.warmup_steps else (2 if step < self.freeze_step else 3)
        self.gamma = min(max((step - self.warmup_steps) / self.ramp_steps, 0.0), 1.0)
        for p in self.parameters():
            p.requires_grad_(self.stage != 3)
        self.M.requires_grad_(self.stage == 2)
        self.rho.requires_grad_(self.stage == 2)

    def forward(self, hidden_states, target, mask=None):
        # [Spatial-DsRA v3]: Uniform independent layer losses; never mix layers or detach H_l.
        if len(hidden_states) != self.num_layers or target.ndim != 3 or target.shape[-1] != self.teacher_dim:
            raise ValueError("Expected L hidden tensors and target [B,N,teacher_dim]")
        if mask is None:
            mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        if mask.shape != target.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Visual mask must be bool [B,N]")
        target = F.normalize(target.detach().float(), dim=-1)
        coeff = self.gamma * self.a_max * self.rho.tanh()
        losses = []
        for layer, hidden in enumerate(hidden_states):
            if hidden.shape != (*target.shape[:2], self.llm_dim):
                raise ValueError(f"Layer {layer}: unexpected visual shape {tuple(hidden.shape)}")
            b = F.gelu(self.down(self.norm(hidden)))
            # [Spatial-DsRA v3]: [B,N,db] -> [B,N,r] -> [B,N,db] -> [B,N,dg].
            b = b + ((b @ self.M) * coeff[layer]) @ self.M.T
            u = self.up(b)
            distance = 1 - (F.normalize(u.float(), dim=-1) * target).sum(-1)
            losses.append(distance.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1))
        layer_losses = torch.stack(losses)
        # [Spatial-DsRA v3]: FP32 regularizers; Stage III excludes both auxiliary regularizers.
        with torch.autocast(device_type=self.M.device.type, enabled=False):
            m = self.M.float()
            orth = (m.T @ m - torch.eye(m.shape[1], device=m.device)).square().sum()
            reg_r = coeff.float().square().mean()
        reg = self.beta_r * reg_r + self.eta * orth if self.stage != 3 else layer_losses.new_zeros(())
        return layer_losses.mean(), reg, layer_losses.detach(), reg_r.detach(), orth.detach()


def teacher_target(features, patch_hw, target_hw):
    # [Spatial-DsRA v3]: Preserve [batch, camera, row, column]; bilinear first, positional encoding second.
    from vggt.heads.utils import create_uv_grid, position_grid_to_embed
    b, views, patches, channels = features.shape
    if patches != math.prod(patch_hw) or min(*patch_hw, *target_hw) <= 0:
        raise ValueError("Teacher patch grid does not match feature count")
    grid = features.detach().float().permute(0, 1, 3, 2).reshape(b * views, channels, *patch_hw)
    grid = F.interpolate(grid, size=target_hw, mode="bilinear", align_corners=True)
    h, w = target_hw
    uv = create_uv_grid(w, h, aspect_ratio=w / h, dtype=grid.dtype, device=grid.device)
    # Retain the base SF positional encoding convention and amplitude.
    pe = position_grid_to_embed(uv, channels).permute(2, 0, 1).unsqueeze(0) * 0.1
    return (grid + pe).reshape(b, views, channels, h * w).permute(0, 1, 3, 2).reshape(b, views * h * w, channels)


def parameter_groups(projector, normal_params, lr, basis_lr_scale):
    # [Spatial-DsRA v3]: Keep M/rho in optimizer from step zero, with separately configured lower LR.
    if not 0 < basis_lr_scale < 1:
        raise ValueError("basis_lr_scale must be explicitly set between 0 and 1")
    ordinary = [p for n, p in projector.named_parameters() if n not in ("M", "rho")]
    return [{"params": list(normal_params) + ordinary, "lr": lr, "lr_scale": 1.0},
            {"params": [projector.M, projector.rho], "lr": lr * basis_lr_scale, "lr_scale": basis_lr_scale}]


class FrozenTeacherCache:
    # [Spatial-DsRA v3]: Cache identity includes teacher weights, selected layer and exact augmented images.
    def __init__(self, directory, checkpoint):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.teacher_id = digest.hexdigest()

    def key(self, sample, layer):
        pixels = sample.detach().cpu().contiguous()
        identity = f"v3:{self.teacher_id}:{layer}:{tuple(pixels.shape)}:{pixels.dtype}".encode()
        return hashlib.sha256(identity + pixels.view(torch.uint8).numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def extract(self, teacher, images, layer):
        outputs = []
        patch_hw = (images.shape[-2] // teacher.patch_size, images.shape[-1] // teacher.patch_size)
        for sample in images:
            sample_id = self.key(sample, layer)
            path = self.directory / (sample_id + ".pt")
            if path.exists():
                record = torch.load(path, map_location="cpu", weights_only=True)
                if tuple(record["patch_hw"]) != patch_hw or record["sample_id"] != sample_id:
                    raise ValueError("Teacher cache metadata mismatch")
                features = record["features"]
            else:
                teacher.eval()
                result = teacher(sample.unsqueeze(0))
                features = result["features"][layer][0, :, result["patch_start_idx"]:, :].detach().to("cpu", torch.bfloat16)
                record = dict(sample_id=sample_id, camera_ids=list(range(sample.shape[0])),
                              features=features, patch_hw=patch_hw, image_hw=tuple(sample.shape[-2:]),
                              teacher_id=self.teacher_id, layer=layer,
                              preprocessing="Exact post-augmentation teacher input; camera-major patch order")
                temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                torch.save(record, temporary)
                os.replace(temporary, path)
            outputs.append(features.to(images.device))
        return torch.stack(outputs), patch_hw


@torch.no_grad()
def extract_teacher(teacher, images, layer):
    # [Spatial-DsRA v3]: Online recomputation is safe for geometric augmentation; cache never crosses coordinates.
    teacher.eval()
    cache = getattr(teacher, "spatial_cache", None)
    if cache is not None:
        return cache.extract(teacher, images, layer)
    result = teacher(images)
    features = result["features"][layer][:, :, result["patch_start_idx"]:, :]
    h, w = result["images"].shape[-2:]
    return features, (h // teacher.patch_size, w // teacher.patch_size)


def spatial_diagnostics(projector, backbone):
    # [Spatial-DsRA v3]: Track layer modulation and LoRA gradient norms without retaining hidden-state gradients.
    values = {"dsra_stage": projector.stage, "dsra_gamma": projector.gamma}
    coeff = (projector.gamma * projector.a_max * projector.rho.detach().float().tanh())
    for i, row in enumerate(coeff):
        values[f"modulation_layer_{i+1}_norm"] = row.norm().item()
    for name in ("M", "rho"):
        grad = getattr(projector, name).grad
        values[f"dsra_{name}_grad_norm"] = 0.0 if grad is None else grad.detach().float().norm().item()
    squared = {}
    for name, parameter in backbone.named_parameters():
        if "lora_" in name and parameter.grad is not None:
            prefix = name.split("self_attn")[0].rstrip(".")
            squared[prefix] = squared.get(prefix, 0) + parameter.grad.detach().float().square().sum()
    values.update({f"lora_grad/{name}": value.sqrt().item() for name, value in squared.items()})
    return values
