"""Seer detector model.

A self-supervised ViT backbone (DINOv3 preferred, DINOv2 fallback) fully
fine-tuned for AI-image detection - "continuation training", the recipe the
current state of the art (Pangram Image) found beats frozen-feature probing:
AI detection is not an ordinary downstream task, so the general-purpose
features are allowed to specialise rather than staying frozen.

Dual heads:
  * global head: [CLS ; mean(patch tokens)] -> MLP -> single real/AI logit
  * local head : per-patch linear -> patch logits (heatmaps / composites)

Alternatively (probe mode), the backbone stays frozen and a single linear
head is trained on features tapped from several blocks of the backbone -
early blocks for high-frequency fingerprints, mid/late blocks for
semantics - one real/AI logit per page (image). See ProbeConfig.

Total parameters with DINOv3 ViT-L: ~302M (15% of the 2B budget).
"""

import inspect
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import PARAM_BUDGET

DEFAULT_BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _tiny_backbone() -> nn.Module:
    """Random tiny ViT for tests / CI - no network access needed."""
    from transformers import Dinov2Config, Dinov2Model

    kwargs = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        image_size=224,
        patch_size=16,
    )
    try:
        cfg = Dinov2Config(**kwargs, num_register_tokens=2)
    except TypeError:  # older transformers without register support
        cfg = Dinov2Config(**kwargs)
    return Dinov2Model(cfg)


def load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    if name == "tiny":
        return _tiny_backbone()
    from transformers import AutoConfig, AutoModel

    if pretrained:
        try:
            return AutoModel.from_pretrained(name)
        except Exception as e:  # gated repo, network error, ...
            raise RuntimeError(
                f"Could not load backbone '{name}': {e}\n"
                "If this is a gated model (DINOv3 is), accept its license on the "
                "model page and authenticate with `hf auth login`, or point "
                "`backbone` at an open checkpoint such as 'facebook/dinov2-large'."
            ) from e
    cfg = AutoConfig.from_pretrained(name)
    return AutoModel.from_config(cfg)


class SeerDetector(nn.Module):
    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        head_dropout: float = 0.1,
        probe_layers: Optional[List[int]] = None,
    ):
        """`probe_layers` (from ProbeConfig.layers) switches to page-level
        linear-probe mode: no dual heads, a single linear layer over
        concatenated multi-block features instead. None (default) builds the
        continuation-training model."""
        super().__init__()
        self.backbone_name = backbone
        self.backbone = load_backbone(backbone, pretrained)
        cfg = self.backbone.config
        self.hidden_size = int(cfg.hidden_size)
        self.patch_size = int(getattr(cfg, "patch_size", 16) or 16)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)

        if probe_layers is not None:
            self.probe = True
            self.probe_layers = self._resolve_probe_layers(probe_layers)
            d = len(self.probe_layers) * 2 * self.hidden_size
            # LayerNorm aligns the very different scales of early vs late
            # block outputs; the map from standardized features to the logit
            # stays linear (a linear probe in the DINOv2/v3 eval sense).
            self.probe_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        else:
            self.probe = False
            self.global_head = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_size),
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(self.hidden_size, 1),
            )
            # Per-patch head: enables heatmaps and composite training.
            self.local_head = nn.Linear(self.hidden_size, 1)

        # DINOv2-style models need explicit position-embedding interpolation
        # when fed a resolution different from pretraining; DINOv3 (RoPE) does not.
        try:
            sig = inspect.signature(self.backbone.forward)
        except (ValueError, TypeError):
            sig = None
        self._interp_pos = sig is not None and "interpolate_pos_encoding" in sig.parameters

    # ------------------------------------------------------------------ utils

    def patch_grid(self, res: int) -> int:
        assert res % self.patch_size == 0, (
            f"resolution {res} must be a multiple of patch size {self.patch_size}"
        )
        return res // self.patch_size

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def budget_report(self) -> str:
        n = self.parameter_count()
        return (
            f"{n:,} parameters ({n / 1e9:.2f}B) "
            f"-> {100.0 * n / PARAM_BUDGET:.1f}% of the 2B budget"
        )

    def enable_gradient_checkpointing(self):
        try:
            self.backbone.gradient_checkpointing_enable()
        except Exception:
            pass

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _resolve_probe_layers(self, layers: List[int]) -> List[int]:
        """Absolute, sorted, de-duplicated block indices from a (possibly
        empty = evenly spaced, possibly negative = from the end) spec."""
        L = self.num_layers
        if not layers:
            layers = [L // 4, L // 2, 3 * L // 4, -1]
        out = []
        for i in layers:
            j = i + L if i < 0 else i
            if not 0 <= j < L:
                raise ValueError(
                    f"probe layer {i} is out of range for a backbone with "
                    f"{L} blocks (valid: 0..{L - 1}, -1 = final block)"
                )
            if j not in out:
                out.append(j)
        return sorted(out)

    # ---------------------------------------------------------------- forward

    def _split_tokens(self, seq: torch.Tensor, H: int, W: int):
        """Returns (cls_token, patch_tokens) from any hidden state.

        transformers versions disagree on whether register tokens appear in
        `last_hidden_state`, so we infer the layout from the expected patch
        count instead of assuming [CLS, registers, patches]."""
        cls = seq[:, 0]
        rest = seq[:, 1:]
        expected = (H // self.patch_size) * (W // self.patch_size)
        n_rest = rest.shape[1]
        if n_rest != expected and self.num_register_tokens > 0:
            # register tokens present (usually right after CLS)
            if n_rest - self.num_register_tokens == expected or n_rest > expected:
                rest = rest[:, self.num_register_tokens:]
        return cls, rest

    def features(self, images: torch.Tensor):
        """Returns (cls_token, patch_tokens) from the final block."""
        kwargs = dict(pixel_values=images)
        if self._interp_pos:
            kwargs["interpolate_pos_encoding"] = True
        out = self.backbone(**kwargs)
        return self._split_tokens(out.last_hidden_state, images.shape[-2], images.shape[-1])

    def layer_features(self, images: torch.Tensor) -> torch.Tensor:
        """Page-level multi-layer features: (B, n_layers * 2 * hidden).

        Per tapped block, [CLS ; mean(patch tokens)] pooled, all blocks
        concatenated. Runs the frozen backbone without grad - a linear probe
        never needs activation gradients through the backbone.
        """
        kwargs = dict(pixel_values=images, output_hidden_states=True)
        if self._interp_pos:
            kwargs["interpolate_pos_encoding"] = True
        H, W = images.shape[-2], images.shape[-1]
        with torch.no_grad():
            out = self.backbone(**kwargs)
            # hidden_states = (embeddings, block_0_out, ..., block_{L-1}_out)
            feats = []
            for idx in self.probe_layers:
                cls, patches = self._split_tokens(out.hidden_states[idx + 1], H, W)
                feats.append(torch.cat([cls, patches.mean(dim=1)], dim=-1))
        return torch.cat(feats, dim=-1)

    def forward(self, images: torch.Tensor) -> dict:
        if self.probe:
            logits = self.probe_head(self.layer_features(images)).squeeze(-1)
            return {"logits": logits, "patch_logits": None}  # page-level only
        cls, patches = self.features(images)
        pooled = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        logits = self.global_head(pooled).squeeze(-1)  # (B,)
        patch_logits = self.local_head(patches).squeeze(-1)  # (B, G*G)
        return {"logits": logits, "patch_logits": patch_logits}


def detection_loss(
    logits: torch.Tensor,
    patch_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    patch_labels: torch.Tensor,
    patch_weight: float = 0.5,
):
    """Image-level BCE + per-patch BCE (the composite-training objective).

    `patch_logits=None` (page-level linear probe) drops the patch term."""
    loss_g = F.binary_cross_entropy_with_logits(logits, labels)
    if patch_logits is None:
        return loss_g, {"loss_global": loss_g.item(), "loss_patch": 0.0}
    loss_p = F.binary_cross_entropy_with_logits(patch_logits, patch_labels)
    return loss_g + patch_weight * loss_p, {"loss_global": loss_g.item(), "loss_patch": loss_p.item()}


# ---------------------------------------------------------------------- EMA


class EMA:
    """Exponential moving average of model weights (state-dict based)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow

    @staticmethod
    def from_state_dict(sd: dict) -> "EMA":
        ema = object.__new__(EMA)
        ema.decay = 1.0
        ema.shadow = sd
        return ema


# ------------------------------------------------------- optimizer / checkpoints


def build_param_groups(model: SeerDetector, base_lr: float, head_lr: float, llrd: float, weight_decay: float):
    """Parameter groups with layer-wise learning-rate decay (ViT fine-tuning
    standard): deeper backbone blocks get LR closer to base_lr, embeddings get
    the most decayed LR, heads get their own (higher) LR."""

    head_params = []
    for head_name in ("global_head", "local_head", "probe_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            head_params += list(head.parameters())
    groups = [
        {"params": [p for p in head_params if p.requires_grad], "lr": head_lr, "name": "head"}
    ]
    backbone = model.backbone
    L = model.num_layers
    layer_params = [[] for _ in range(L)]
    emb_params, top_params = [], []

    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        m = re.match(r"encoder\.layer\.(\d+)\.", n)
        if m and 0 <= int(m.group(1)) < L:
            layer_params[int(m.group(1))].append(p)
        elif n.startswith("embeddings") or "pos_embed" in n or "pos_embeds" in n:
            emb_params.append(p)
        else:
            top_params.append(p)

    emb_lr = base_lr * (llrd**L)
    if emb_params:
        groups.append({"params": emb_params, "lr": emb_lr, "name": "embeddings"})
    for i, ps in enumerate(layer_params):
        if ps:
            lr_i = base_lr * (llrd ** (L - 1 - i))
            groups.append({"params": ps, "lr": lr_i, "name": f"layer_{i}"})
    if top_params:
        groups.append({"params": top_params, "lr": base_lr, "name": "backbone_top"})
    for g in groups:
        g["weight_decay"] = weight_decay
    return groups


def save_checkpoint(path, model: SeerDetector, cfg, step: int, optimizer=None, scheduler=None, ema: Optional[EMA] = None, metrics: Optional[dict] = None):
    from .config import config_to_dict

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
            "metrics": metrics or {},
            "train_cfg": config_to_dict(cfg),
            "backbone_name": model.backbone_name,
            "param_count": model.parameter_count(),
        },
        path,
    )


def load_checkpoint(path, device="cpu", prefer_ema: bool = True) -> tuple:
    """Returns (model, cfg_dict, ckpt)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("train_cfg") or {}
    backbone = cfg_dict.get("backbone") or ckpt.get("backbone_name") or DEFAULT_BACKBONE
    probe_cfg = cfg_dict.get("probe") or {}
    probe_layers = list(probe_cfg.get("layers") or []) if probe_cfg.get("enabled") else None
    model = SeerDetector(backbone, pretrained=False, probe_layers=probe_layers)
    sd = ckpt.get("ema") if (prefer_ema and ckpt.get("ema")) else None
    if sd is None:
        sd = ckpt["model"]
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, cfg_dict, ckpt
