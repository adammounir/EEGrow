"""Training loop with growth: gradient descent + periodic growth.

Key point: instead of fully resetting the optimizer after each growth (which loses
momentum across the WHOLE network), we preserve the optimizer state (Adam momentum)
of the unchanged parameters and only start fresh for the weights that grew.
"""

from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from gromo.utils.training_utils import compute_statistics, evaluate_model

# The "input_extension_scaling is null" warning only fires at s=0 (the no-growth
# baseline of the line search): benign, verified empirically. Silence it.
warnings.filterwarnings("ignore", message=".*input_extension_scaling is null.*")


@dataclass
class TrainConfig:
    total_epochs: int = 150
    grow_every: int = 15
    lr: float = 6.25e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    log_every: int = 10


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_optimizer(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


def rebuild_optimizer_preserving_state(
    old_opt: torch.optim.Optimizer, model: torch.nn.Module, cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, int]:
    """Rebuild the optimizer after a growth, keeping momentum of unchanged params.

    After ``apply_change``, gromo replaces the weight tensors that grew with NEW
    ``Parameter`` objects; the others (BN, classifier...) stay the SAME objects. We
    transfer the state for those identical objects and leave fresh state for the
    new/enlarged params.

    Returns
    -------
    (optimizer, number_of_params_whose_momentum_was_transferred)
    """
    old_state = old_opt.state  # dict keyed by Parameter object
    new_opt = make_optimizer(model, cfg)
    transferred = 0
    for p in model.parameters():
        if p in old_state:  # same object => unchanged param => keep its momentum
            new_opt.state[p] = old_state[p]
            transferred += 1
    return new_opt, transferred


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_epoch(model, loader, opt, crit, device) -> float:
    model.train()
    tot = 0.0
    for x, y in loader:
        opt.zero_grad()
        loss = crit(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / len(loader)


def grow_step(model, train_loader, device) -> None:
    """One gromo growth: stats -> optimal update -> line search -> apply_change.

    NB: ``compute_optimal_updates`` relies on ``eigh``, which is NOT implemented on
    MPS -- growth therefore requires a CPU device.
    """
    crit_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    crit_mean = torch.nn.CrossEntropyLoss(reduction="mean")
    model.set_growing_layers(index=0)
    compute_statistics(model, train_loader, loss_function=crit_sum, device=device)
    model.compute_optimal_updates()
    model.reset_computation()
    model.dummy_select_update()
    best_loss, best_s = float("inf"), 0.0
    for s in (0.0, 0.1, 0.5, 1.0):
        model.set_scaling_factor(s)
        loss, _ = evaluate_model(model, train_loader, crit_mean,
                                 use_extended_model=True, device=device)
        if loss < best_loss:
            best_loss, best_s = loss, s
    model.set_scaling_factor(best_s)
    model.apply_change()


def run_model(name, model, train_loader, test_loader, *, grow: bool,
              cfg: TrainConfig, device, log_fn=print) -> dict:
    """Train a model (with or without growth) and return the history."""
    crit = torch.nn.CrossEntropyLoss()
    hist = {"epoch": [], "test_acc": [], "params": [], "name": name}
    log_fn(f"\n--- {name}: start width={model.growable_width}, "
           f"{count_params(model):,} params ---")
    t0 = time.time()
    opt = make_optimizer(model, cfg)
    for epoch in range(1, cfg.total_epochs + 1):
        train_epoch(model, train_loader, opt, crit, device)
        if grow and epoch % cfg.grow_every == 0 and epoch < cfg.total_epochs:
            grow_step(model, train_loader, device)
            opt, kept = rebuild_optimizer_preserving_state(opt, model, cfg)
        acc = accuracy(model, test_loader, device)
        hist["epoch"].append(epoch)
        hist["test_acc"].append(acc)
        hist["params"].append(count_params(model))
        if epoch % cfg.log_every == 0 or epoch == 1:
            log_fn(f"  [{name:>10}] epoch {epoch:>3}  test_acc={acc:.3f}  "
                   f"width={model.growable_width}  "
                   f"params={count_params(model):,}")
    hist["seconds"] = time.time() - t0
    hist["final_params"] = count_params(model)
    log_fn(f"  [{name}] done in {hist['seconds']:.0f}s -- "
           f"final acc {hist['test_acc'][-1]:.3f}, best {max(hist['test_acc']):.3f}, "
           f"final width {model.growable_width}, "
           f"{count_params(model):,} params")
    return hist
