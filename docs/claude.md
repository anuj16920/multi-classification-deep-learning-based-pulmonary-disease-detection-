# CLAUDE.md

# MEDXPERT Development Instructions

You are working on a research-grade machine learning project called MEDXPERT.

Read these files before making architectural or experimental decisions:

1. CONTENT.md
2. PROJECT_DETAILS.md

These files define the research contract.

---

# 1. PRIMARY OBJECTIVE

Implement MEDXPERT as described in the published abstract.

The implementation must contain:

- Multi-expert learning
- Adaptive disease routing
- Adaptive expert memory
- Graph-guided collaboration
- Multi-class classification
- Uncertainty estimation
- Explainability
- Calibration evaluation

The final objective is to reproduce the published results as closely as scientifically possible.

---

# 2. DO NOT FABRICATE

This is extremely important.

NEVER:

- hard-code metrics
- fabricate results
- manipulate predictions
- manipulate evaluation datasets
- remove difficult test samples
- tune on test data
- alter confusion matrices
- manually edit ROC results
- claim an implementation detail came from the paper if it was not provided
- generate fake training logs

All metrics must be generated from actual model predictions.

---

# 3. RESEARCH TRACEABILITY

Every major implementation decision must be traceable.

When implementing something not explicitly specified by the abstract, document:

IMPLEMENTATION DECISION

Reason:

Alternative considered:

Potential effect on reproducibility:

Example:

"The abstract does not specify the graph layer. We use GAT as an implementation choice because adaptive attention provides a natural mechanism for disease-to-disease knowledge transfer."

Do not write:

"The paper uses GAT"

unless verified.

---

# 4. ARCHITECTURAL REQUIREMENTS

The model must contain:

INPUT
↓
FEATURE ENCODER
↓
ADAPTIVE ROUTER
↓
MULTIPLE EXPERTS
↓
ADAPTIVE MEMORY
↓
GRAPH COLLABORATION
↓
CLASSIFICATION

Auxiliary:

UNCERTAINTY
EXPLAINABILITY

---

# 5. CODE QUALITY

Use:

Python 3.x

PyTorch

Torchvision

timm

PyTorch Geometric where required

scikit-learn

OpenCV

Albumentations

NumPy

Pandas

Matplotlib

Seaborn

Captum or equivalent explainability library

Prefer:

- typed Python
- dataclasses
- modular classes
- clear docstrings
- configuration-driven experiments
- deterministic seeds where practical

Avoid:

- huge monolithic files
- duplicated code
- hard-coded paths
- hard-coded hyperparameters
- hidden global state

---

# 6. PROJECT STRUCTURE

Use:

configs/
data/
models/
training/
evaluation/
explainability/
inference/
experiments/
scripts/
tests/
outputs/
checkpoints/
logs/
docs/

---

# 7. CONFIGURATION

Do not scatter hyperparameters throughout source code.

Use YAML.

Example:

configs/default.yaml

configs/medxpert.yaml

configs/baselines/resnet50.yaml

configs/baselines/densenet121.yaml

etc.

---

# 8. DATASET

Before training:

VERIFY:

- total image count
- class count
- class distribution
- train count
- test count
- duplicates
- corrupted images
- image dimensions
- file formats

Expected:

10,000 total

2,500/class

8,000 train

2,000 test

If actual data differs, STOP and report the discrepancy.

Do not silently modify the dataset.

---

# 9. DATA SPLITTING

Never split after augmentation.

Never allow augmented versions of the same image across train and test.

Use deterministic splitting.

Save split manifests.

Example:

data/splits/train.csv

data/splits/test.csv

The manifests become immutable experiment inputs.

---

# 10. MODEL DEVELOPMENT ORDER

Implement in this order:

STEP 1
Dataset pipeline

STEP 2
Baseline models

STEP 3
Shared encoder

STEP 4
Expert blocks

STEP 5
Adaptive router

STEP 6
Expert memory

STEP 7
Graph collaboration

STEP 8
Fusion

STEP 9
Uncertainty

STEP 10
Explainability

STEP 11
Full training

STEP 12
Ablations

STEP 13
Final evaluation

Do not attempt to debug the entire system simultaneously.

---

# 11. BASELINES

Implement:

ResNet50
DenseNet121
EfficientNet-B3
ViT
Swin Transformer

All baselines must use:

- same dataset split
- documented preprocessing
- documented training protocol
- reproducible seeds

---

# 12. MEDXPERT

Required conceptual modules:

SharedEncoder

DiseaseRouter

DiseaseExpert

ExpertMemory

DiseaseGraph

GraphCollaboration

FusionHead

UncertaintyEstimator

ExplainabilityEngine

---

# 13. ROUTER

The router must be input-dependent.

Minimum:

router_logits = Router(shared_features)

routing_weights = softmax(router_logits)

expert_features = experts(shared_features)

weighted_features = weighted_sum(
    routing_weights,
    expert_features
)

The router must remain differentiable.

---

# 14. MEMORY

Memory must participate in forward computation.

Do not create an unused prototype dictionary.

Memory should influence expert representation or classification.

Memory updates must be clearly defined.

---

# 15. GRAPH

Graph must participate in forward computation.

Nodes:

disease experts

Node features:

expert representations

Graph output:

graph-enhanced expert representations

The graph cannot merely exist as a visualization.

---

# 16. LOSS

Start with a simple scientifically defensible objective.

Initial:

CrossEntropyLoss

Then experimentally evaluate whether auxiliary objectives improve the architecture.

Possible auxiliary losses:

prototype consistency

routing regularization

graph consistency

Only add losses when there is a clear reason.

Document every addition.

---

# 17. TRAINING

Training must support:

- mixed precision
- checkpointing
- early stopping
- learning-rate scheduling
- gradient clipping
- logging
- resume training

Use AMP where supported.

Save:

best checkpoint

last checkpoint

training history

configuration

---

# 18. GPU

Automatically detect:

CUDA

MPS

CPU

Print hardware information at startup.

Do not assume a specific GPU.

---

# 19. METRICS

Compute:

Accuracy

Precision

Recall

F1

ROC-AUC

ECE

Brier Score

Confusion Matrix

Per-class metrics

Use explicit averaging methods.

For multiclass metrics, document:

macro

weighted

micro where appropriate

AUC:

one-vs-rest

macro

per-class

---

# 20. REPRODUCIBILITY

Every run should save:

seed

config

git commit

dataset manifest hash

environment information

model parameter count

training duration

hardware

metrics

checkpoint path

---

# 21. TESTING

Create unit tests for:

dataset loading

label mapping

router

expert shapes

memory updates

graph layer

model forward pass

loss

metrics

checkpoint loading

Grad-CAM

Run tests before expensive training.

---

# 22. DEBUG MODE

Create a tiny dataset/debug mode.

Example:

--debug

It should:

- use tiny dataset
- use few samples
- run 1–2 epochs
- verify complete pipeline
- generate outputs

This allows architecture debugging before expensive training.

---

# 23. EXPERIMENT MANAGEMENT

Every experiment gets a unique ID.

Example:

EXP-001

Store:

outputs/EXP-001/

    config.yaml
    metrics.json
    metrics.csv
    predictions.csv
    training.log
    confusion_matrix.png
    roc.png
    calibration.png

---

# 24. RESULT COMPARISON

Create a central:

outputs/experiment_registry.csv

Columns:

experiment_id

model

seed

dataset_version

accuracy

precision

recall

f1

auc

ece

brier

checkpoint

git_commit

timestamp

---

# 25. TARGET RESULTS

Published:

97.35 Accuracy

97.35 Precision

97.35 Recall

97.34 F1

99.89 AUC

These are NOT constants.

Use them only for comparison.

---

# 26. IF RESULTS ARE LOWER

Do not cheat.

Investigate systematically:

1. Dataset mismatch
2. Split mismatch
3. Preprocessing mismatch
4. Label issues
5. Backbone
6. Expert architecture
7. Router
8. Memory
9. Graph
10. Training hyperparameters
11. Loss
12. Seed variance

Run controlled experiments.

---

# 27. IF RESULTS ARE HIGHER

Do not assume success.

Check:

- leakage
- duplicate images
- test contamination
- incorrect preprocessing
- accidental test tuning

Then report actual results.

---

# 28. EXPLAINABILITY

Generate:

original image

predicted class

probabilities

Grad-CAM

overlay

Save examples.

---

# 29. UNCERTAINTY

Provide:

confidence

entropy

calibrated confidence

uncertainty category

Do not claim clinical certainty.

---

# 30. MEDICAL DISCLAIMER

The application is a research prototype.

Never state:

"This patient has TB."

Use:

"The model predicts TB with X confidence."

---

# 31. COMMUNICATION STYLE

When reporting progress:

Say exactly what works.

Example:

"Dataset pipeline verified: 10,000 images, 2,500/class."

Do not say:

"Everything is perfect."

unless verified.

---

# 32. BEFORE MAJOR CHANGES

Inspect the existing code.

Do not rewrite working components unnecessarily.

Prefer incremental changes.

Maintain backward compatibility with saved experiment outputs where practical.

---

# 33. FINAL REQUIREMENT

The final repository should allow a researcher to run:

dataset preparation

baseline training

MEDXPERT training

evaluation

ablation experiments

uncertainty analysis

explainability generation

without manually editing source code.