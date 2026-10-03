"""Model sanity check: tiny debug batch through every component.

This is an ENGINEERING verification, NOT evidence of model performance.
Checks: forward, backward, router gradient, expert bank, memory update,
graph gradient, fusion, classifier, Grad-CAM hook.
"""
from __future__ import annotations
import argparse, json, sys
import torch

from medxpert.utils import load_config, set_seed, pick_device
from medxpert.models import build_model
from medxpert.models.medxpert import MEDXPERT

from _common import apply_debug_overrides


def _check(name: str, ok: bool, note: str = "") -> dict:
    print(("[OK]  " if ok else "[FAIL]") + f" {name}" + (f" — {note}" if note else ""))
    return {"name": name, "ok": bool(ok), "note": note}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/medxpert.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    apply_debug_overrides(cfg)
    set_seed(int(cfg["project"]["seed"]))
    device = pick_device()
    print(f"[sanity] device={device} config={args.config}")

    results = []
    n_classes = len(cfg["classes"])
    model = build_model(cfg, n_classes=n_classes).to(device)
    is_medx = isinstance(model, MEDXPERT)
    H = int(cfg["data"]["image_size"])
    x = torch.randn(4, 3, H, H, device=device)
    y = torch.randint(0, n_classes, (4,), device=device)

    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    aux = out[1] if isinstance(out, tuple) else None
    results.append(_check("forward_shape", logits.shape == (4, n_classes),
                          f"{tuple(logits.shape)}"))

    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    results.append(_check("backward", any(p.grad is not None and p.grad.abs().sum() > 0
                                          for p in model.parameters())))

    if is_medx:
        router_grad = any(p.grad is not None for p in model.router.parameters())
        results.append(_check("router_gradient", router_grad))
        expert_grad = any(p.grad is not None for p in model.experts.parameters())
        results.append(_check("expert_gradients", expert_grad))
        if model.graph is not None:
            graph_grad = (model.graph.adj_logits.grad is not None)
            results.append(_check("graph_gradient", graph_grad))
        if model.memory is not None:
            # Trigger an EMA update and confirm initialized flips.
            agg = (aux.expert_feats * aux.routing.unsqueeze(-1)).sum(dim=1)
            model.memory.update(agg.detach(), y)
            results.append(_check("memory_update", bool(model.memory.initialized.any())))

        # Dict output path.
        d = model(x, return_dict=True)
        for k in ("logits", "probabilities", "predicted_class", "confidence",
                  "routing_weights"):
            results.append(_check(f"dict_output[{k}]", k in d))

    # Grad-CAM wiring on the current model (uses last Conv2d).
    try:
        from medxpert.explainability import GradCAM
        cam = GradCAM(model)
        heatmap, pred, probs = cam(x[:1].detach().clone().requires_grad_(True))
        cam.close()
        results.append(_check("gradcam_hook", heatmap.ndim == 2))
    except Exception as e:  # noqa: BLE001
        results.append(_check("gradcam_hook", False, str(e)))

    print(json.dumps({"results": results,
                      "all_ok": all(r["ok"] for r in results)}, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
