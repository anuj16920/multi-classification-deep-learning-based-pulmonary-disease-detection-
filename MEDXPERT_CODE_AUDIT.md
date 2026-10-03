# MEDXPERT — Code Audit (Hardening Pass)

Date: 2026-10-01
Scope: targeted corrections to the existing MEDXPERT scaffold at
`D:/5th international anujj ones/medxpert/`. Architecture is preserved.

---

## 1. Architecture (unchanged, verified wired)

```
CXR
  └─> SharedEncoder (configurable backbone, fail-hard on init error)
        └─> DiseaseRouter (soft adaptive routing; top-k optional)
              └─> ExpertBank (N parameter-efficient blocks)
                    └─> ExpertMemory (EMA/learnable prototypes)       [ablatable]
                          └─> DiseaseGraph (GAT/GCN, learned adj.)    [ablatable]
                                └─> FusionHead (attention/weighted/concat)
                                      └─> ClassificationHead → logits
                                      └─> ExpertSpecialization → aux logits
                                      └─> Uncertainty (val-fitted temperature)
                                      └─> Explainability (Grad-CAM)
```

Component status is reported programmatically by
`MEDXPERT.component_status()` and saved per experiment in
`experiment_manifest.json`.

Classes (canonical ordering, locked in `src/medxpert/utils/classes.py`):

```
0  Normal
1  Tuberculosis
2  COVID-19
3  Pneumonia
```

Expert-to-class mapping defaults to the identity (`expert_i ↔ class_i`) and
can be overridden in YAML under `model.experts.mapping`.

---

## 2. Corrections (what changed)

### Research integrity

| # | Problem | Fix | File |
|---|---|---|---|
| 1 | Silent `timm → TinyCNN` fallback hid backbone-init failures | `BackboneInitError` on any backbone that is not literally `tiny_debug`. TinyCNN only when opted-in; emits warning. | `src/medxpert/models/encoder.py` |
| 2 | Test loader was being used for validation | Trainer takes `train_loader` + `val_loader` only. `scripts/train_medxpert.py` builds only `("train","val")`. | `src/medxpert/training/trainer.py`, `scripts/train_medxpert.py` |
| 3 | Temperature scaling fit on test | `evaluate.py` fits temperature on `val_logits`/`val_labels`, freezes, then applies to test. Static leakage check in tests. | `scripts/evaluate.py`, `tests/test_no_test_leakage.py` |
| 4 | No validation split existed | New train/val/test splitter with `validation_fraction` (default 0.10). Carves test first (locked), then val from the dev pool, train is remainder. | `src/medxpert/data/splits.py` |
| 5 | No split-metadata or integrity audit | Writes `split_metadata.json`, `manifest.sha256`, per-split CSVs. `check_split_disjoint()` runs in `train_medxpert.py` before training. | `src/medxpert/data/splits.py`, `scripts/train_medxpert.py` |
| 6 | No duplicate detection | Optional `image_hash` column + `duplicate_report()` writing `duplicate_report.csv`. | `src/medxpert/data/splits.py`, `scripts/prepare_data.py` |
| 7 | Debug and official modes could blur together | `cfg.debug.enabled` flag; official training refuses to run when it's true unless the operator passes `--i-am-debugging`. Debug artifacts also carry `debug_mode: true` in their experiment manifest. | `scripts/train_medxpert.py`, `scripts/_common.py`, `configs/default.yaml` |
| 8 | Class mapping was duplicated across files | Single source of truth `src/medxpert/utils/classes.py`. All config loads call `validate_class_mapping(cfg)` and raise on mismatch. | `src/medxpert/utils/classes.py`, `data/dataset.py`, `models/medxpert.py`, `scripts/*` |

### Model wiring

| # | Problem | Fix |
|---|---|---|
| 9 | No expert-to-class mapping was explicit | `EXPERT_TO_CLASS` constant + `validate_expert_mapping(cfg)` + optional YAML override. |
| 10 | No expert specialization objective | New `expert_specialization` loss (BCE on per-expert class indicators). Head is always built; the loss is weighted at `0.05` in `medxpert.yaml` and `0.0` or omitted in ablations. Guards for `n_experts != n_classes` → no-op. |
| 11 | Memory could crash on NaN/empty batches | Safeguards: NaN features dropped, empty-class skipped, NaN-after-update rolled back. `diagnostics()` returns norm/drift/usage counts. |
| 12 | Graph had no stability guards or inspection | NaN-guard between layers; `diagnostics()` returns adjacency + in/out degree; `evaluate.py` saves `adjacency.npy` + `graph_diagnostics.json`. |
| 13 | Forward output was opaque for analysis | `forward(x, return_dict=True)` returns a structured dict (`logits`, `probabilities`, `predicted_class`, `confidence`, `routing_weights`, `expert_features`, `memory_features`, `graph_features`, `adjacency`, `expert_specialization_logits`, `class_names`, `expert_to_class`). The training loop still consumes the lightweight `(logits, aux)` tuple. |
| 14 | No routing analysis | `evaluation/routing_analysis.py` + `evaluate.py` writes `routing.csv` and includes `routing_report` in `metrics.json` (per-class mean routing, expert utilization, mean routing entropy, top-1 alignment). |

### Reproducibility & auditing

| # | Fix |
|---|---|
| 15 | `experiment_manifest.json` written per experiment with seed, config path, checkpoint path, test manifest path, timestamp, git commit, debug flag, component status. |
| 16 | `write_run_bundle()` already saved `config.yaml`, `git_commit.txt`, `env.json`, `hardware.json`, `manifest_hash.txt`, `seed.txt`. Still active, now joined by `experiment_manifest.json`. |
| 17 | Trainer saves `val_predictions.npz` so post-hoc calibration fitting is reproducible from disk. |
| 18 | `scripts/preflight.py` runs class/expert-mapping validation, hardware detection, raw-dir counts, split disjointness, model forward+backward. Non-zero exit on any critical failure. |
| 19 | `scripts/sanity_check.py` runs the full component wiring (forward, backward, router grad, expert grad, graph grad, memory update, dict output, Grad-CAM hook) on a debug batch — explicitly labeled an engineering check, not a performance claim. |

### Baselines (preserved)

Baseline models remain `encoder → head` and never touch router/memory/graph.
Verified by `BaselineClassifier` (`src/medxpert/models/baselines.py`) and by
the baseline configs in `configs/baselines/`.

### Ablations (preserved)

The 10 ablation configs in `configs/ablations/` each flip concrete config
switches (`model.memory.enabled`, `model.graph.enabled`,
`model.router.top_k`, loss list). `tests/test_component_effect.py` verifies
that disabling memory / graph genuinely removes their contribution from the
forward output (`memory_features`/`graph_features` become `None`).

---

## 3. Research integrity — how test leakage is prevented

- `src/medxpert/training/trainer.py` has no symbol named `test_loader`. The
  static check `tests/test_no_test_leakage.py` asserts this.
- `scripts/train_medxpert.py` builds `("train","val")` loaders only; a static
  regex assertion in the leakage test enforces it.
- `scripts/evaluate.py` always does:
  `_collect_logits(..., val_loader)` → `temp_scaler.fit(val_logits, val_labels)`
  → `_collect_logits(..., test_loader)` → apply scaler → write metrics.
  Order check in `test_no_test_leakage.py` enforces this.
- Test manifest is written once and only read for final evaluation; the
  pipeline refuses to run if `check_split_disjoint` finds any overlap.

---

## 4. Reproducibility — what every run records

Per experiment under `outputs/<EXP_ID>/`:

```
config.yaml                   # resolved config, including debug flag
git_commit.txt
seed.txt
env.json                      # python/platform
hardware.json                 # cuda name/version
manifest_hash.txt             # sha256 of split manifest
history.json                  # per-epoch train+val metrics
training.log                  # trainer log (stderr tee)
best.pt / last.pt             # weights
val_predictions.npz           # frozen val logits+labels (for calibration refits)
metrics.json / metrics.csv    # final test metrics (incl. val_ece, val_brier)
classification_report.{csv,txt}
per_class_metrics.csv
confusion_matrix.{csv,png}
roc_curves.png / pr_curves.png / calibration_curve.png
prediction_results.csv        # per-sample label/pred/probs
routing.csv                   # per-sample routing weights (MEDXPERT only)
adjacency.npy                 # learned graph adjacency (MEDXPERT only)
graph_diagnostics.json        # (MEDXPERT only)
memory_state.pt / memory_diagnostics.json  (MEDXPERT only)
experiment_manifest.json      # single-file traceability bundle
explainability/explain_XXX.png
```

Plus one row per experiment in `outputs/experiment_registry.csv`.

---

## 5. Component verification (empirical, this session)

Debug run (`EXP-DEBUG`, synthetic 16-images-per-class dataset):

| Component | Verified by | Status |
|---|---|---|
| Router | `test_router.py`, dict output `routing_weights` populated, routing.csv written | ACTIVE |
| Experts | `test_experts.py`, `test_component_effect.py` | ACTIVE |
| Memory | `test_memory.py`, `test_memory_safety.py`, memory_state.pt + diagnostics saved | ACTIVE |
| Graph | `test_graph.py`, `test_component_effect.py`, adjacency.npy saved | ACTIVE |
| Fusion | model forward test | ACTIVE |
| Uncertainty | `test_uncertainty.py`, temperature fit on val, exported in metrics.json | ACTIVE |
| Explainability | `test_gradcam.py`, 4 PNGs produced | ACTIVE |

`pytest tests/` → **51 passed** in ~52s.
`scripts/sanity_check.py` → `all_ok: true`.
`scripts/preflight.py --skip-dataset` → `hard_fail: false`.

---

## 6. Remaining limitations (honest)

- Dataset NOT included. The `data/raw/` tree is empty. Everything above was
  verified on synthetic debug images (16 per class) under
  `%TEMP%/medxpert_debug/EXP-DEBUG/`.
- No training on the real Kaggle `Tuberculosis-tb-chest-xray-dataset` has
  been performed in this hardening pass. Published metrics (97.35% / 99.89%)
  remain target references only; they are never written as computed metrics.
- Full-size backbones (ConvNeXt-tiny, ResNet50, …) are compatible but their
  initial pretrained weights are downloaded by timm on first use. If offline,
  set `pretrained: false` in the YAML or pre-stage the timm cache.
- The ablation runner (`scripts/run_ablations.py`) iterates ablation YAMLs
  and invokes train+evaluate; it has not been re-verified end-to-end this
  pass but each ablation config has been exercised individually.
- Grad-CAM targets the last `nn.Conv2d` by default. For transformer backbones
  (ViT, Swin), pass an explicit target module.

---

## 7. Dataset readiness

Dataset not yet included. Code is ready for dataset integration via:

```powershell
# 1. Place real images under data/raw/{Normal,Tuberculosis,COVID-19,Pneumonia}/
# 2. Build the official 10k / 2k split with validation carved out of the dev pool:
python scripts/prepare_data.py --config configs/default.yaml --seed 42

# 3. Pre-flight BEFORE spending GPU time:
python scripts/preflight.py --config configs/medxpert.yaml

# 4. Train MEDXPERT:
python scripts/train_medxpert.py --config configs/medxpert.yaml --exp-id EXP-010

# 5. Final evaluation on the untouched test set:
python scripts/evaluate.py --exp-id EXP-010
```

Do NOT claim the published 97.35% / 99.89% numbers until a real training
run on the verified dataset actually produces them. The pipeline does not
and will not fabricate those numbers.
