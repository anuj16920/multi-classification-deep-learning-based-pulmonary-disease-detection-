from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .encoder import SharedEncoder
from .router import DiseaseRouter
from .experts import ExpertBank
from .memory import ExpertMemory
from .graph import DiseaseGraph
from .fusion import FusionHead
from .head import ClassificationHead
from ..utils.classes import (
    CLASS_NAMES, validate_class_mapping, validate_expert_mapping,
)


@dataclass
class ForwardAux:
    """Auxiliary outputs for loss functions and diagnostics.

    This is a lightweight bag of references — nothing is cloned, so a training
    step does not pay for tensors it does not use. Deeper analysis uses the
    model's forward(..., return_dict=True) path.
    """
    routing: torch.Tensor                        # (B, N) soft routing weights
    expert_feats: torch.Tensor                   # (B, N, D) post-expert, pre-memory
    memory_feats: Optional[torch.Tensor]         # (B, N, D) post-memory
    graph_feats: Optional[torch.Tensor]          # (B, N, D) post-graph
    fused: torch.Tensor                          # (B, out_dim)
    expert_logits: Optional[torch.Tensor] = None # (B, N) specialization head output


class MEDXPERT(nn.Module):
    """Graph-Guided Collaborative Multi-Expert Framework with Adaptive Expert Memory.

    Design (soft adaptive routing — not hard conditional computation):

        CXR
         |
        Shared Encoder (configurable backbone; fails hard on init error)
         |
        Adaptive Disease Router  -> soft (B,N) weights over experts
         |
        Expert Bank              -> (B,N,D) one vector per expert
         |          (model.memory.enabled)
        Expert Memory            -> cosine-attention over class prototypes
         |          (model.graph.enabled)
        Disease Graph (GAT/GCN)  -> message passing over experts
         |
        Fusion Head              -> (B, out_dim), weighted by routing
         |
        Classification Head      -> (B, C) logits
         |
        [optional] Expert specialization logits (B, N) — each expert gets a
        1-d score that an auxiliary loss (losses.expert_specialization) can
        push toward its assigned class. Mapping lives in utils.classes.

    Ablation switches (preserved from the original design):
        model.memory.enabled = false  -> skip memory module entirely
        model.graph.enabled  = false  -> skip graph module entirely
        model.router.top_k   = 0      -> soft routing; >0 enables top-k

    Forward returns (logits, aux). Pass return_dict=True for a verbose
    structured output suitable for analysis / explainability without changing
    the training loop API.
    """

    def __init__(self, cfg: Any, n_classes: int):
        super().__init__()
        validate_class_mapping(cfg)
        self.n_classes = n_classes
        self.class_names: List[str] = list(CLASS_NAMES)
        self.expert_to_class: Dict[int, str] = validate_expert_mapping(cfg)

        m = cfg["model"]
        enc = m["encoder"]
        exp = m["experts"]
        rt = m["router"]
        mem = m["memory"]
        gr = m["graph"]
        fus = m["fusion"]
        head = m["head"]

        self.encoder = SharedEncoder(
            backbone=enc["backbone"],
            pretrained=bool(enc.get("pretrained", True)),
            out_dim=int(enc.get("out_dim", 0)) or None,
        )
        shared_dim = self.encoder.out_dim

        self.n_experts = int(exp["n_experts"])
        if self.n_experts != n_classes:
            # Not strictly fatal (ablations may collapse experts), but mapping
            # only makes sense when the counts match. We note it in the aux
            # output so losses can skip specialization safely.
            self._expert_mapping_valid = (self.n_experts == len(self.expert_to_class))
        else:
            self._expert_mapping_valid = True

        self.experts = ExpertBank(
            n_experts=self.n_experts,
            in_dim=shared_dim,
            hidden_dim=int(exp["hidden_dim"]),
            out_dim=int(exp["hidden_dim"]),
            dropout=float(exp.get("dropout", 0.1)),
        )
        expert_dim = self.experts.out_dim

        self.router = DiseaseRouter(
            in_dim=shared_dim,
            n_experts=self.n_experts,
            hidden_dim=int(rt.get("hidden_dim", 128)),
            temperature=float(rt.get("temperature", 1.0)),
            top_k=int(rt.get("top_k", 0)),
        )

        self.memory_enabled = bool(mem.get("enabled", True))
        if self.memory_enabled:
            self.memory = ExpertMemory(
                n_classes=n_classes,
                feat_dim=expert_dim,
                prototypes_per_class=int(mem.get("prototypes_per_class", 1)),
                momentum=float(mem.get("momentum", 0.9)),
                update_mode=str(mem.get("update_mode", "ema")),
            )
        else:
            self.memory = None

        self.graph_enabled = bool(gr.get("enabled", True))
        if self.graph_enabled:
            self.graph = DiseaseGraph(
                n_nodes=self.n_experts,
                in_dim=expert_dim,
                hidden_dim=int(gr.get("hidden_dim", 192)),
                heads=int(gr.get("heads", 4)),
                n_layers=int(gr.get("n_layers", 2)),
                layer=str(gr.get("layer", "gat")),
                edge_init=str(gr.get("edge_init", "learned")),
            )
        else:
            self.graph = None

        self.fusion = FusionHead(
            in_dim=expert_dim,
            n_experts=self.n_experts,
            out_dim=int(fus.get("out_dim", 256)),
            mode=str(fus.get("mode", "attention")),
        )
        self.head = ClassificationHead(
            in_dim=self.fusion.out_dim,
            n_classes=n_classes,
            dropout=float(head.get("dropout", 0.2)),
        )

        # Expert specialization: each expert projects its own vector to a
        # single scalar. The auxiliary loss then matches this scalar to a
        # one-hot indicator of whether that expert's assigned class is the
        # ground truth. Trainable always; a loss weight of 0 (default in most
        # configs) renders it inert in training.
        self.expert_specialization = nn.Linear(expert_dim, 1)

    # ------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, Optional[ForwardAux]] | Dict[str, Any]:
        shared = self.encoder(x)                                 # (B, D)
        routing = self.router(shared)                            # (B, N)
        experts_out = self.experts(shared)                       # (B, N, D)
        feats = experts_out
        memory_feats = None
        if self.memory is not None:
            memory_feats = self.memory(feats)
            feats = memory_feats
        graph_feats = None
        if self.graph is not None:
            graph_feats = self.graph(feats)
            feats = graph_feats
        fused = self.fusion(feats, routing)                      # (B, out_dim)
        logits = self.head(fused)                                # (B, C)
        expert_logits = self.expert_specialization(feats).squeeze(-1)  # (B, N)
        aux = ForwardAux(
            routing=routing,
            expert_feats=experts_out,
            memory_feats=memory_feats,
            graph_feats=graph_feats,
            fused=fused,
            expert_logits=expert_logits,
        )
        if not return_dict:
            return logits, aux

        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred,
            "confidence": conf,
            "routing_weights": routing,
            "expert_features": experts_out,
            "memory_features": memory_feats,
            "graph_features": graph_feats,
            "adjacency": (self.graph.adjacency().detach()
                          if self.graph is not None else None),
            "expert_specialization_logits": expert_logits,
            "class_names": self.class_names,
            "expert_to_class": self.expert_to_class,
        }

    # --------------------------------------------------------- memory step

    @torch.no_grad()
    def update_memory(self, aux: ForwardAux, labels: torch.Tensor) -> None:
        """Call after optimizer.step(). No-op if memory is disabled or in
        learnable mode (prototype learning handled by backprop in that mode)."""
        if self.memory is None:
            return
        agg = (aux.expert_feats * aux.routing.unsqueeze(-1)).sum(dim=1)
        self.memory.update(agg, labels)

    # --------------------------------------------------- component status

    def component_status(self) -> Dict[str, bool]:
        return {
            "encoder": True,
            "router": True,
            "experts": True,
            "memory": self.memory is not None,
            "graph": self.graph is not None,
            "fusion": True,
            "classifier": True,
            "expert_specialization": True,
        }
