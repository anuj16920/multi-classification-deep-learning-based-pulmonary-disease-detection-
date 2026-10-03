# MEDXPERT — Project Details

## Project Type

Research-grade deep learning system for multi-class chest X-ray classification.

## Research Domain

Computer Vision
Medical Imaging
Deep Learning
Multi-Expert Systems
Graph Neural Networks
Explainable AI
Uncertainty Estimation

---

# Core Problem

Different thoracic diseases can exhibit visually overlapping patterns in chest X-ray images.

The system aims to improve discriminative representation learning by using multiple disease experts instead of relying only on a conventional single end-to-end classifier.

---

# Target Classes

| ID | Class |
|---|---|
| 0 | Normal |
| 1 | Tuberculosis |
| 2 | COVID-19 |
| 3 | Pneumonia |

Class IDs may be changed if required by dataset conventions, but the mapping MUST be stored in a single configuration file and remain consistent across training and evaluation.

---

# Dataset

Target dataset:

Tuberculosis-tb-chest-xray-dataset

Target experimental subset:

10,000 images

Balanced:

2,500 images/class

Training:

8,000

Testing:

2,000

Expected:

| Class | Train | Test | Total |
|---|---:|---:|---:|
| TB | 2,000 | 500 | 2,500 |
| COVID-19 | 2,000 | 500 | 2,500 |
| Pneumonia | 2,000 | 500 | 2,500 |
| Normal | 2,000 | 500 | 2,500 |

---

# Architecture

## High-Level

Input image

↓

Image preprocessing

↓

Shared feature representation

↓

Adaptive disease router

↓

Disease experts

↓

Expert memory

↓

Graph-guided collaboration

↓

Prediction head

↓

Prediction + uncertainty + explanation

---

# Shared Encoder

The shared encoder produces a feature representation from the CXR.

The exact architecture is an implementation decision unless specified by the complete published paper.

Candidate architecture choices should be evaluated experimentally.

Possible candidates:

- ConvNeXt
- EfficientNet
- ResNet
- DenseNet
- hybrid CNN-transformer

Do not claim a particular candidate was used in the published work unless verified.

---

# Disease Experts

Four conceptual disease experts:

- TB Expert
- COVID Expert
- Pneumonia Expert
- Normal Expert

Each expert receives shared representations and produces a disease-aware feature representation.

The experts should not be implemented as four independent full-scale models unless there is a strong computational justification.

Prefer parameter-efficient expert blocks where appropriate.

---

# Adaptive Router

Input:

shared feature vector

Output:

expert routing weights

Example:

TB: 0.65
COVID: 0.12
Pneumonia: 0.20
Normal: 0.03

The router must be differentiable.

Possible implementation:

Router MLP

↓

softmax

↓

expert weights

The implementation should support top-k routing as an optional experiment.

---

# Expert Memory

Memory maintains disease prototypes.

Each class has one or more prototype vectors.

Basic prototype:

P_c

Update:

P_c(t+1) = alpha * P_c(t) + (1-alpha) * F_c

where:

P_c = prototype
F_c = current class representation
alpha = memory momentum

The implementation should support:

- EMA prototype update
- learnable prototype memory
- multiple prototypes/class as an experimental extension

Only one method should be used in the final primary configuration and documented clearly.

---

# Graph

Graph nodes:

TB
COVID-19
Pneumonia
Normal

Node features:

expert representations / memory-enhanced representations

Edges:

learned or initialized disease relationships

Graph operation:

message passing between expert nodes

Possible implementation:

Graph Attention Network

or

Graph Convolutional Network

The final choice must be treated as an implementation decision unless supported by the complete paper.

---

# Graph Output

The graph-enhanced expert features are fused before classification.

Possible fusion:

weighted sum

concatenation + projection

attention fusion

The selected mechanism must be documented.

---

# Classification Head

Input:

graph-enhanced representation

Output:

4-class logits

Activation:

softmax for inference

Loss:

cross entropy as initial default unless a more suitable experimentally validated objective is required.

---

# Uncertainty

Primary objective:

Identify low-confidence predictions.

Possible methodology:

Predictive entropy:

H(p) = -Σ p_i log(p_i)

Calibration:

Temperature scaling

Metrics:

ECE
Brier Score
Reliability diagram

The final methodology must be documented.

---

# Explainability

Primary method:

Grad-CAM or compatible class activation visualization.

Required outputs:

Original image

Predicted class

Confidence

Heatmap

Overlay

---

# Training

Initial training should be reproducible and configurable.

All parameters must live in YAML configuration files.

Do not hard-code training parameters throughout Python files.

---

# Baseline Models

Implement independently:

ResNet50

DenseNet121

EfficientNet-B3

Vision Transformer

Swin Transformer

All baselines must use the same dataset split.

Where practical, preprocessing and training budgets should be comparable.

---

# Evaluation

Final evaluation must generate:

metrics.json

metrics.csv

classification_report.csv

confusion_matrix.png

roc_curves.png

pr_curves.png

calibration_curve.png

per_class_metrics.csv

prediction_results.csv

explainability examples

---

# Experiment Naming

Use:

EXP-001
EXP-002
EXP-003
...

Each experiment must have:

configuration
metrics
checkpoint
logs
timestamp
git commit
hardware information

---

# Target Results

Published target:

Accuracy = 97.35%

Precision = 97.35%

Recall = 97.35%

F1 = 97.34%

AUC = 99.89%

These are target values, not constants.

---

# Success Criteria

A successful implementation must:

1. Run end-to-end.
2. Reproduce dataset composition.
3. Train all baselines.
4. Train MEDXPERT.
5. Perform ablations.
6. Calculate all metrics.
7. Generate uncertainty outputs.
8. Generate explainability outputs.
9. Preserve reproducibility.
10. Produce results that can be compared with the published abstract.