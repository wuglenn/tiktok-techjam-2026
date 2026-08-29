"""Muon (MomentUm Orthogonalized by Newton-Schulz) + auxiliary AdamW.

2D hidden weights get Muon; embeddings, norms, biases, and other non-matrix
parameters keep AdamW. Learning rates are Moonshot-scaled (`match_rms_adamw`)
so the existing AdamW LRs / LLRD schedule stay in the right ballpark.

Refs:
  Keller Jordan, https://kellerjordan.github.io/posts/muon/
  Moonshot Moonlight, https://github.com/MoonshotAI/Moonlight
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch


NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Zeroth-power / orthogonalize a 2D (or batched 2D) matrix via quintic NS."""
    if G.ndim < 2:
        raise ValueError("Newton-Schulz expects a matrix")
    a, b, c = NS_COEFFS
    # bf16 NS is the point on GPU; CPU tests stay in fp32.
    dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """SGD-momentum, then orthogonalize. Conv filters are viewed as 2D."""
    momentum.lerp_(grad, 1.0 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    shape = update.shape
    if update.ndim >= 3:
        update = update.reshape(update.size(0), -1)
    return newton_schulz5(update, steps=ns_steps).reshape(shape)


def _adjust_lr(lr: float, shape, fn: str) -> float:
    if len(shape) < 2:
        return lr
    a, b = int(shape[0]), int(math.prod(shape[1:]))
    if fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(a, b))
    if fn in (None, "original"):
        return lr * math.sqrt(max(1.0, a / max(b, 1)))
    raise ValueError(f"unknown muon adjust_lr_fn {fn!r}")


def _use_muon(param: torch.nn.Parameter, group_name: str) -> bool:
    """Hidden 2D (+ hidden conv) weights only. Embeddings stay on AdamW."""
    if group_name == "embeddings":
        return False
    return param.ndim >= 2


def _split_groups(groups: Iterable[dict]) -> List[dict]:
    out = []
    for raw in groups:
        g = dict(raw)
        if "use_muon" in g:
            out.append(g)
            continue
        name = g.get("name", "")
        muon_ps, adam_ps = [], []
        for p in g["params"]:
            (muon_ps if _use_muon(p, name) else adam_ps).append(p)
        shared = {k: v for k, v in g.items() if k != "params"}
        if muon_ps:
            out.append({**shared, "params": muon_ps, "use_muon": True})
        if adam_ps:
            out.append({**shared, "params": adam_ps, "use_muon": False})
    return out


def _as_groups(params) -> List[dict]:
    if params and isinstance(params, (list, tuple)) and isinstance(params[0], dict):
        return [dict(g) for g in params]
    return [{"params": list(params)}]


@torch.no_grad()
def _adamw_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    b1, b2 = betas
    exp_avg.lerp_(grad, 1.0 - b1)
    exp_avg_sq.lerp_(grad.square(), 1.0 - b2)
    m = exp_avg / (1.0 - b1**step)
    v = exp_avg_sq / (1.0 - b2**step)
    return m / (v.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Single-device Muon + auxiliary AdamW.

    Param groups may set `use_muon` explicitly; otherwise 2D non-embedding
    tensors are assigned to Muon and everything else to AdamW. Extra group
    keys (e.g. `name` from `build_param_groups`) are kept.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0e-5,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        adjust_lr_fn: str = "match_rms_adamw",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
            use_muon=False,
        )
        super().__init__(_split_groups(_as_groups(params)), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            if group.get("use_muon"):
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-_adjust_lr(lr, p.shape, group["adjust_lr_fn"]))
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adamw_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss


def build_optimizer(groups, cfg) -> torch.optim.Optimizer:
    name = str(getattr(cfg, "optimizer", "muon") or "muon").lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    if name != "muon":
        raise ValueError(f"unknown optimizer {name!r}; expected 'muon' or 'adamw'")
    return Muon(
        groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=getattr(cfg, "muon_momentum", 0.95),
        ns_steps=getattr(cfg, "muon_ns_steps", 5),
        adjust_lr_fn=getattr(cfg, "muon_adjust_lr", "match_rms_adamw"),
    )


def group_param_counts(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    muon = adam = 0
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        if g.get("use_muon"):
            muon += n
        else:
            adam += n
    return muon, adam
