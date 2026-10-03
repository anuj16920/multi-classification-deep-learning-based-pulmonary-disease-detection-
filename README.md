# MEDXPERT

**Graph-Guided Collaborative Multi-Expert Framework with Adaptive Expert Memory**
for explainable 4-class chest X-ray classification (Normal / TB / COVID-19 / Pneumonia).

Research-grade reproduction scaffold. The classes, components and target metrics are
specified in `../Project_details.md`, `../context.md`, `../claude.md` (research contract).

> **Status.** This repository is a complete, test-covered scaffold. It runs end-to-end
> in `--debug` mode on synthetic images without any dataset download, so you can verify
> every component wires correctly on your laptop. The real dataset (10,000 CXRs,
> 2,500/class) is **not** included — see "Dataset" below.
>
> **No metrics are fabricated.** Nothing in this repo claims to have reproduced the
> published 97.35% / 99.89% numbers. Those numbers are present only as TARGETS in
> configs and reports, and every metric that gets written is computed from actual
> model predictions.

---

## 1. Install

```powershell
# Python 3.10+ recommended. From the medxpert/ directory:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

GPU is optional. Everything runs on CPU in debug mode. Real training wants CUDA.

## 2. Verify the install (no dataset needed)

```powershell
# Unit tests — exercises dataset pipeline, router, experts, memory, graph, model,
# metrics, Grad-CAM, calibration, encoder fail-hard, no-test-leakage, component
# effect, memory safety, class mapping. All synthetic data; no download.
pytest tests/ -v

# Model-wiring sanity check (forward/backward on every component, Grad-CAM hook).
python scripts/sanity_check.py --config configs/medxpert.yaml

# End-to-end dry run on synthetic images (16 synth imgs/class → 9/3/4 split, 2 epochs).
python scripts/train_medxpert.py --config configs/medxpert.yaml --exp-id EXP-DEBUG --debug
python scripts/evaluate.py --exp-id EXP-DEBUG --debug
python scripts/explain.py  --exp-id EXP-DEBUG --debug --n 4
```

If this all works, the pipeline is wired. Nothing above says anything about
how well the model classifies real X-rays — that depends on the real dataset.

## 3. Dataset

Target: **Tuberculosis-tb-chest-xray-dataset** on Kaggle (search that exact name).

Expected final layout under `data/raw/`:

```
data/raw/
├── Normal/        2500 images (.png or .jpg)
├── Tuberculosis/  2500
├── COVID-19/      2500
└── Pneumonia/     2500
```

Build the official splits (train / **val** / test, with sha256 manifest, split
metadata, and duplicate report):

```powershell
python scripts/prepare_data.py --config configs/default.yaml --seed 42
```

The script will **STOP** and report rather than silently proceed if class counts
or totals don't match the contract (10,000 total, 2,500/class, 8,000 development
pool, 2,000 test). Validation is carved out of the development pool at the
`data.validation_fraction` ratio (default 0.10): per class, this gives
1800 train + 200 val + 500 test. The 2,000-image test set is locked once
written.

Pre-flight the pipeline before spending GPU hours:

```powershell
python scripts/preflight.py --config configs/medxpert.yaml
```

Non-zero exit if any check fails (missing data, overlapping splits, backbone
init error, model forward/backward failure).

## 4. Train baselines (one-by-one)

```powershell
python scripts/train_baseline.py --config configs/baselines/resnet50.yaml       --exp-id EXP-001
python scripts/train_baseline.py --config configs/baselines/densenet121.yaml    --exp-id EXP-002
python scripts/train_baseline.py --config configs/baselines/efficientnet_b3.yaml --exp-id EXP-003
python scripts/train_baseline.py --config configs/baselines/vit.yaml            --exp-id EXP-004
python scripts/train_baseline.py --config configs/baselines/swin.yaml           --exp-id EXP-005
```

## 5. Train MEDXPERT + ablations

```powershell
python scripts/train_medxpert.py  --config configs/medxpert.yaml         --exp-id EXP-010
python scripts/run_ablations.py   --config configs/medxpert.yaml --out-root outputs/
```

Ablation grid (`configs/ablations/`):
1. `base.yaml` — shared encoder + classifier only
2. `router.yaml` — + adaptive router over experts (no memory, no graph)
3. `memory.yaml` — + expert memory
4. `graph.yaml` — + graph collaboration
5. `router_memory.yaml`
6. `router_graph.yaml`
7. `memory_graph.yaml`
8. `full.yaml` — full MEDXPERT
9. `full_uncertainty.yaml` — + temperature scaling + entropy
10. `full_explain.yaml` — + Grad-CAM batch export

## 6. Evaluate

```powershell
python scripts/evaluate.py --exp-id EXP-010
```

Writes to `outputs/EXP-010/`:

```
config.yaml
metrics.json         # accuracy, precision, recall, f1 (macro+weighted), auc (ovr+macro+per-class)
metrics.csv          # same, flat
classification_report.csv
per_class_metrics.csv
confusion_matrix.png
roc_curves.png
pr_curves.png
calibration_curve.png
prediction_results.csv
training.log
```

And appends one row to `outputs/experiment_registry.csv`.

## 7. Explainability

```powershell
python scripts/explain.py --exp-id EXP-010 --n 16
```

Writes `outputs/EXP-010/explainability/` with original / predicted class /
confidence / Grad-CAM / overlay per sample.

## 8. Repo layout

```
configs/        YAML — all hyperparameters live here, nothing hard-coded in .py
src/medxpert/
    data/       dataset, splits, transforms, verification, synthetic debug images
    models/     shared encoder, router, experts, memory, graph, fusion, head, medxpert.py, baselines.py
    training/   trainer, losses, schedulers
    evaluation/ metrics, calibration, plots
    explainability/ gradcam
    uncertainty/    predictive entropy + temperature scaling
    experiments/    registry.csv manager
    utils/      seed, hardware (CUDA/MPS/CPU auto), config loader
scripts/        prepare_data / train_baseline / train_medxpert / evaluate / explain / run_ablations
tests/          unit tests (synthetic, fast)
outputs/        per-experiment artifacts + experiment_registry.csv
docs/           implementation decisions, reproducibility notes, medical disclaimer
```

## 9. Research integrity rules enforced in code

- No hard-coded metrics; every metric is computed from actual predictions.
- `data/verify.py` raises on class-count / split-count mismatch (no silent fix-up).
- Split manifests are hashed; a hash mismatch at train time is an error.
- Every run writes `config.yaml`, `git_commit.txt`, `env.json`, `hardware.json`,
  `manifest_hash.txt`, `seed.txt`. These are the reproducibility bundle.
- Implementation decisions (encoder choice, graph layer type, memory update rule,
  fusion mechanism) are documented in `docs/implementation_decisions.md`, never
  attributed to the paper.
- `uncertainty/` reports confidence/entropy/calibration. The dashboard phrasing
  is "the model predicts X with Y confidence", never "the patient has X".

## 10. Medical disclaimer

Research prototype. Not a clinical device. Clinical interpretation requires a
licensed professional.
