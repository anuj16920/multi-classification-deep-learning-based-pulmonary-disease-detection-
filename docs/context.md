# MEDXPERT — Project Content & Research Contract

## 1. Project Title

Graph-Guided Multi-Expert Network for Intelligent Disease Monitoring and Machine Learning-Based Chest Disease Diagnosis

## 2. Model Name

MEDXPERT

Full name:

Graph-Guided Collaborative Multi-Expert Framework with Adaptive Expert Memory

## 3. Research Objective

Develop and implement MEDXPERT for explainable multi-class chest X-ray classification involving:

1. Tuberculosis (TB)
2. COVID-19
3. Pneumonia
4. Normal

The implementation must faithfully realize the technical concepts stated in the published abstract.

The objective is to reproduce the published experimental results as closely as possible through a reproducible implementation.

---

# 4. Published Abstract

The published abstract states:

"The Multi-Class Classification of diseases on thoracic area using Chest X-Ray (CXR) Images is difficult owing to visual similarities between the different diseases such as Tuberculosis (TB), COVID-19, and Pneumonia. In the existing studies, the use of End-to-end Deep Learning Networks has been made for classifying the different diseases. Such an approach would restrict the extraction of discriminatory features between the classes. In order to overcome this limitation, in this paper, MEDXPERT (Graph-Guided Collaborative Multi-Expert Framework with Adaptive Expert Memory) for explainable multi-class chest X-ray diagnosis is proposed.

The concept in MEDXPERT lies in the fact that adaptive disease routing is used to select the right experts on the basis of input images. Experts learn discriminative disease features whereas the adaptive expert memory evolves disease prototypes. Graph-guided collaboration helps in modeling the relationship between disease classes and knowledge transfer among experts prior to prediction. Uncertainty estimation helps in identifying low confidence predictions while Explainability provides visual attention maps and diagnostic evidence.

The evaluation of MEDXPERT is done on the publicly available Tuberculosis-tb-chest-xray-dataset, which consists of CXR images from the categories of tuberculosis, COVID-19, pneumonia, and normal.

A dataset having 10,000 CXR images with a balanced number of images in each category, where 2,500 images are selected from each class and 8,000 images are used for training while the remaining 2,000 images are for testing.

The performance of MEDXPERT is assessed in terms of Accuracy, Precision, Recall, F1-score, AUC, and calibration metrics, where ResNet50, DenseNet121, EfficientNet-B3, Vision Transformer, and Swin Transformer have been taken into consideration as baseline approaches.

MEDXPERT obtains an Accuracy of 97.35%, Precision of 97.35%, Recall of 97.35%, F1-score of 97.34%, and AUC of 99.89%."

---

# 5. Non-Negotiable Research Claims

The implementation MUST contain the following concepts:

## 5.1 Multi-Expert Architecture

The system must contain multiple disease-specialized experts.

Experts must learn discriminative disease representations.

The system must not simply be a conventional single-backbone classifier renamed MEDXPERT.

---

## 5.2 Adaptive Disease Routing

The model must contain an explicit routing mechanism.

The router receives information derived from the input CXR and determines expert relevance.

The router should produce expert weights or an equivalent adaptive expert-selection mechanism.

The routing must be input-dependent.

Do NOT implement a static fixed routing table.

---

## 5.3 Adaptive Expert Memory

The system must maintain disease-related prototype representations.

The memory must evolve during training.

The implementation must support updating disease prototypes based on learned representations.

The memory must be integrated into the model rather than existing only as an unused data structure.

---

## 5.4 Graph-Guided Collaboration

Disease experts must participate in a graph-based collaboration mechanism.

The graph represents relationships among disease classes/experts.

The graph mechanism must enable information or knowledge transfer between expert representations before final prediction.

The graph cannot merely be created for visualization.

It must participate in forward computation.

---

## 5.5 Uncertainty Estimation

The model must produce confidence/uncertainty information.

The system should identify low-confidence predictions.

Calibration must be evaluated.

Possible metrics include:

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- reliability diagrams
- predictive entropy

The final implementation must document which uncertainty/calibration methodology is actually used.

---

## 5.6 Explainability

The system must provide visual explanation.

At minimum:

- class prediction
- confidence
- attention/activation visualization

Grad-CAM or an architecture-compatible equivalent may be used.

The implementation must save explainability visualizations for evaluation examples.

---

# 6. Dataset Requirements

The published experiment specifies:

Total images:

10,000

Classes:

- TB
- COVID-19
- Pneumonia
- Normal

Class balance:

2,500 images per class

Train:

8,000 images

Test:

2,000 images

Expected distribution:

2,000 training images per class

500 test images per class

The implementation MUST verify this distribution programmatically.

Never silently train on an imbalanced or incorrectly split dataset.

---

# 7. Data Leakage Prevention

Strictly prevent:

- train/test overlap
- duplicate images across splits
- patient-level leakage where patient identifiers are available
- augmentation leakage
- test-set usage during training
- test-set usage for hyperparameter tuning

The test set must remain isolated until final evaluation.

If patient identifiers are unavailable, explicitly document this limitation.

---

# 8. Baseline Models

Implement and evaluate:

1. ResNet50
2. DenseNet121
3. EfficientNet-B3
4. Vision Transformer
5. Swin Transformer

These are BASELINE approaches.

Do NOT describe these architectures as the internal MEDXPERT architecture unless explicitly justified and documented.

---

# 9. MEDXPERT Components

Required conceptual pipeline:

CXR
→ feature representation
→ adaptive disease router
→ disease experts
→ adaptive expert memory
→ graph-guided collaboration
→ prediction

Additional outputs:

prediction
confidence
uncertainty
explainability

---

# 10. Target Published Results

The published abstract reports:

Accuracy: 97.35%

Precision: 97.35%

Recall: 97.35%

F1-score: 97.34%

AUC: 99.89%

These numbers are TARGET REPRODUCTION VALUES.

They must NEVER be hard-coded into evaluation output.

They must NEVER be fabricated.

The actual implementation must calculate metrics from actual model predictions.

If reproduction differs from the published results, report the actual result and investigate the cause.

---

# 11. Important Research Integrity Rule

Never modify predictions to reach the published metrics.

Never:

- manually edit predictions
- remove difficult samples after evaluation
- tune on the test set
- fabricate metrics
- hard-code target metrics
- alter confusion matrices
- alter ROC curves
- report unpublished results as published results

The purpose is genuine reproduction.

---

# 12. Unknown Details

The published abstract does NOT specify:

- exact CNN backbone used inside MEDXPERT
- exact expert architecture
- exact routing equation
- exact graph architecture
- exact graph type
- exact memory implementation
- exact loss
- optimizer
- learning rate
- batch size
- epochs
- scheduler
- augmentation configuration
- random seed
- exact uncertainty method

Therefore Claude Code MUST NOT claim that any particular implementation detail is "from the paper" unless the full paper/source explicitly provides it.

Implementation decisions must be documented as:

"Implementation choice required to operationalize the published abstract."

---

# 13. Research Experiments

Required experiments:

### Baselines

- ResNet50
- DenseNet121
- EfficientNet-B3
- ViT
- Swin Transformer

### MEDXPERT Ablations

1. Base expert architecture
2. + Adaptive Router
3. + Expert Memory
4. + Graph Collaboration
5. + Router + Memory
6. + Router + Graph
7. + Memory + Graph
8. Full MEDXPERT
9. Full MEDXPERT + uncertainty
10. Full MEDXPERT + explainability

---

# 14. Required Metrics

Classification:

- Accuracy
- Precision
- Recall
- F1-score

Per-class metrics must also be reported.

AUC:

- one-vs-rest ROC-AUC
- macro AUC
- per-class AUC

Calibration:

- ECE
- Brier Score
- reliability diagram

Additional:

- confusion matrix
- ROC curves
- precision-recall curves
- inference time
- parameter count
- FLOPs where practical

---

# 15. Reproducibility

Every experiment must record:

- random seed
- dataset version
- train/test split
- model configuration
- optimizer
- learning rate
- batch size
- number of epochs
- scheduler
- loss
- checkpoint path
- git commit hash
- Python version
- PyTorch version
- CUDA version
- GPU information

---

# 16. Medical Safety

This project is a research prototype.

It must NOT be represented as a clinically validated diagnostic device.

Outputs must be described as:

"model prediction"

and not as:

"confirmed medical diagnosis."

The interface should clearly state that professional clinical interpretation is required.

---

# 17. Expected Deliverables

The final repository should contain:

- dataset preparation scripts
- preprocessing pipeline
- baseline implementations
- MEDXPERT implementation
- training scripts
- evaluation scripts
- ablation scripts
- uncertainty module
- explainability module
- configuration files
- trained checkpoint management
- metrics JSON/CSV
- confusion matrices
- ROC plots
- calibration plots
- Grad-CAM visualizations
- experiment logs
- README
- reproducibility documentation
- research report