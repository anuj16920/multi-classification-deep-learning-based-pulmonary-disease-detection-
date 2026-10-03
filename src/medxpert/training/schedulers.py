from __future__ import annotations
from typing import Any, Tuple
import math
import torch


def build_optimizer_scheduler(model, cfg: Any, steps_per_epoch: int):
    tr = cfg["training"]
    opt_name = str(tr.get("optimizer", "adamw")).lower()
    lr = float(tr.get("lr", 3e-4))
    wd = float(tr.get("weight_decay", 0.01))
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer {opt_name!r}")

    epochs = int(tr.get("epochs", 1))
    warmup_epochs = int(tr.get("warmup_epochs", 0))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    sched_name = str(tr.get("scheduler", "cosine")).lower()

    if sched_name == "none":
        sched = None
    elif sched_name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler {sched_name!r}")
    return opt, sched
