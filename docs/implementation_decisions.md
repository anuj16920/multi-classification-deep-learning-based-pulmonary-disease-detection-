# Implementation decisions

The published abstract does NOT specify many architectural details. The choices
below are implementation decisions required to operationalize it. None of them
should be attributed to the paper.

## Shared encoder — `convnext_tiny` (default)
**Reason.** Strong ImageNet-pretrained performance at modest parameter count;
fits a 4GB laptop GPU with image size 224.
**Alternatives considered.** ResNet50, DenseNet121, EfficientNet-B3, ViT-B/16,
Swin-T — all implemented as *baselines* for comparison, not inside MEDXPERT.
**Effect on reproducibility.** Replacing the backbone will shift absolute metrics
by a few points. For a direct reproduction attempt, override
`model.encoder.backbone` in `configs/medxpert.yaml` and re-train.

## Experts — parameter-efficient MLP blocks
**Reason.** The paper's phrasing ("experts learn discriminative disease features")
is consistent with specialized feature projections. Four full backbones would be
wasteful on a laptop budget and is not required by the text.
**Alternative considered.** Four independent backbones (rejected on compute grounds).

## Router — single-layer MLP + softmax (optional top-k)
**Reason.** Minimal differentiable gating mechanism consistent with "adaptive
disease routing". Soft routing by default; top-k is wired as an optional
experiment via `model.router.top_k`.

## Expert memory — EMA prototypes (default)
**Reason.** Non-parametric; momentum = 0.9 is a conventional starting point for
feature-space prototype tracking.
**Alternative.** `update_mode: learnable` makes prototypes `nn.Parameter`s
co-trained with the loss. Switchable in YAML.

## Graph — GAT with learned adjacency
**Reason.** Attention-based message passing provides a natural, data-driven
expert-to-expert knowledge transfer mechanism — a reasonable operationalization
of "graph-guided collaboration".
**Alternative.** GCN with the same learned adjacency, selectable via
`model.graph.layer: gcn`.

## Fusion — attention over expert vectors
**Reason.** Lets the final representation re-weight experts based on their mutual
information after the graph step; `weighted_sum` and `concat_proj` are also
implemented and selectable.

## Uncertainty — predictive entropy + temperature scaling
**Reason.** The lightest, most defensible calibration method; fit on a held-out
subset after training. Set `cfg.uncertainty.method = "none"` to disable.

## Loss — cross entropy + optional auxiliaries
**Reason.** Start simple. Auxiliary objectives (router entropy, prototype
consistency) are off-by-default (weight 0) in base configs and enabled in the
full MEDXPERT config so the ablation grid can isolate their contribution.
