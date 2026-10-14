# MEDXPERT Benchmark & Ablation Study Results

**Repository**: MEDXPERT (Chest X-ray 4-Class Classifier: Normal / Tuberculosis / COVID-19 / Pneumonia)  
**Hardware Platform**: NVIDIA GeForce GTX 1080 Ti (11,264 MiB VRAM), Driver 535.288.01, CUDA 12.2  
**PyTorch Version**: 2.5.1+cu121 | Python 3.12.3  
**Report Updated**: October 14, 2026  

---

## 1. Research Integrity & Dataset Split Verification

All metrics presented in this document originate directly from real model evaluations on the real test split (`data/splits/manifest.csv`) via `scripts/evaluate.py`. No synthetic data, mock metrics, or debug settings were used.

- **Split Manifest**: `data/splits/manifest.csv`
- **Computed SHA-256**: `19539e98f0bdbb9a4e9a8eccbbc9a3d2dc6fb11e887329ecb7aa1478cd993b60`
- **Reference Hash (`outputs/EXP-010/manifest_hash.txt`)**: `19539e98f0bdbb9a4e9a8eccbbc9a3d2dc6fb11e887329ecb7aa1478cd993b60`
- **Split Hash Status**: **IDENTICAL / MATCH CONFIRMED**

### Split Sample Composition
| Split | Normal | Tuberculosis | COVID-19 | Pneumonia | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,800 | 1,800 | 1,800 | 1,800 | **7,200** |
| **Validation** | 200 | 200 | 200 | 200 | **800** |
| **Test** | 500 | 500 | 500 | 500 | **2,000** |
| **Total** | 2,500 | 2,500 | 2,500 | 2,500 | **10,000** |

### Common Training Budget Across All Experiments (Matching EXP-010)
- **Image Resolution**: 224 × 224 (RGB, ImageNet normalized: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- **Batch Sizes**: Train = 32, Validation/Test = 64
- **Optimizer**: AdamW (`lr: 3e-4`, `weight_decay: 0.01`)
- **LR Schedule**: Cosine annealing with 2 linear warmup epochs
- **Epoch Budget**: Maximum 30 epochs
- **Early Stopping**: Patience = 7 epochs monitoring validation `f1_macro` (mode: max)
- **Gradient Clipping**: Maximum norm 1.0
- **Precision**: Mixed Precision (FP16 AMP enabled)
- **Random Seed**: 42
- **Data Augmentation**: Horizontal flip (p=0.5), Random rotation (±10°), Color jitter (brightness=0.1, contrast=0.1)
- **Calibration**: Temperature scaling fitted strictly on the validation set

---

## 2. Baseline Models Benchmark Table

Test set metrics computed via `scripts/evaluate.py` from `outputs/<EXP_ID>/metrics.json` and `history.json`:

| Model | EXP id | Accuracy % | Precision % (macro) | Recall % (macro) | F1 % (macro) | AUC % (ovr macro) | ECE (test) | Best Epoch | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50** | EXP-001 | 97.45 | 97.47 | 97.45 | 97.45 | 99.79 | 0.0148 | 27 | 43m 27s |
| **DenseNet-121** | EXP-002 | 97.45 | 97.46 | 97.45 | 97.45 | 99.81 | 0.0156 | 22 | 47m 25s |
| **EfficientNet-B3** | EXP-003 | 97.65 | 97.65 | 97.65 | 97.65 | 99.81 | 0.0141 | 13 | 76m 58s |
| **ViT-B/16** | EXP-004 | 95.90 | 95.91 | 95.90 | 95.90 | 99.61 | 0.0156 | 14 | 229m 49s |
| **Swin-T** | EXP-005 | 97.30 | 97.34 | 97.30 | 97.31 | 99.81 | 0.0148 | 22 | 67m 03s |
| **MEDXPERT (Full)** *(Reference)* | EXP-010 | 97.15 | 97.14 | 97.15 | 97.15 | 99.81 | 0.0192 | 23 | 94m 28s |

---

## 3. MEDXPERT Ablation Study Table

Evaluating the specific contribution of individual architectural components (Router, Prototype Memory, Disease Graph) built upon the ConvNeXt-Tiny shared backbone:

| Model Variant | Modules Active (Router / Memory / Graph) | EXP id | Status | Accuracy % | Precision % (macro) | Recall % (macro) | F1 % (macro) | AUC % (ovr macro) | ECE (test) | Best Epoch | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base (ConvNeXt-T)** | None (Encoder + Head only) | EXP-A01 | **Completed** | **97.55** | **97.56** | **97.55** | **97.55** | **99.82** | **0.0185** | 30 | 92m 43s |
| **+ Router** | Router only | EXP-A02 | **Queued / Next** | — | — | — | — | — | — | — | — |
| **+ Memory** | Memory only | EXP-A03 | Queued | — | — | — | — | — | — | — | — |
| **+ Graph** | Graph only | EXP-A04 | Queued | — | — | — | — | — | — | — | — |
| **+ Router + Memory** | Router + Memory | EXP-A05 | Queued | — | — | — | — | — | — | — | — |
| **+ Router + Graph** | Router + Graph | EXP-A06 | Queued | — | — | — | — | — | — | — | — |
| **+ Memory + Graph** | Memory + Graph | EXP-A07 | Queued | — | — | — | — | — | — | — | — |
| **Full MEDXPERT** *(Reference)* | Router + Memory + Graph | EXP-010 | **Completed** | 97.15 | 97.14 | 97.15 | 97.15 | 99.81 | 0.0192 | 23 | 94m 28s |

---

## 4. Per-Class Test Performance Breakdown

All support counts: 500 test images per disease class (2,000 total).

### 4.1 ResNet-50 (`EXP-001`)
- **Normal**: Precision 93.98%, Recall 96.80%, F1 95.37%, AUC 99.62%
- **Tuberculosis**: Precision 100.00%, Recall 100.00%, F1 100.00%, AUC 100.00%
- **COVID-19**: Precision 99.40%, Recall 98.60%, F1 99.00%, AUC 99.97%
- **Pneumonia**: Precision 96.52%, Recall 94.40%, F1 95.45%, AUC 99.59%

### 4.2 DenseNet-121 (`EXP-002`)
- **Normal**: Precision 94.86%, Recall 96.00%, F1 95.43%, AUC 99.63%
- **Tuberculosis**: Precision 99.80%, Recall 100.00%, F1 99.90%, AUC 100.00%
- **COVID-19**: Precision 99.20%, Recall 98.60%, F1 98.90%, AUC 99.97%
- **Pneumonia**: Precision 95.97%, Recall 95.20%, F1 95.58%, AUC 99.65%

### 4.3 EfficientNet-B3 (`EXP-003`)
- **Normal**: Precision 95.43%, Recall 96.00%, F1 95.71%, AUC 99.64%
- **Tuberculosis**: Precision 100.00%, Recall 100.00%, F1 100.00%, AUC 100.00%
- **COVID-19**: Precision 99.00%, Recall 99.20%, F1 99.10%, AUC 99.98%
- **Pneumonia**: Precision 96.17%, Recall 95.40%, F1 95.78%, AUC 99.62%

### 4.4 Vision Transformer (ViT-B/16, `EXP-004`)
- **Normal**: Precision 93.50%, Recall 92.00%, F1 92.74%, AUC 99.35%
- **Tuberculosis**: Precision 100.00%, Recall 99.40%, F1 99.70%, AUC 100.00%
- **COVID-19**: Precision 97.60%, Recall 97.60%, F1 97.60%, AUC 99.72%
- **Pneumonia**: Precision 92.56%, Recall 94.60%, F1 93.57%, AUC 99.38%

### 4.5 Swin Transformer (Swin-T, `EXP-005`)
- **Normal**: Precision 93.27%, Recall 97.00%, F1 95.10%, AUC 99.63%
- **Tuberculosis**: Precision 100.00%, Recall 100.00%, F1 100.00%, AUC 100.00%
- **COVID-19**: Precision 99.39%, Recall 98.20%, F1 98.79%, AUC 99.98%
- **Pneumonia**: Precision 96.71%, Recall 94.00%, F1 95.33%, AUC 99.63%

### 4.6 Base ConvNeXt-Tiny (`EXP-A01`)
- **Normal**: Precision 95.06%, Recall 96.20%, F1 95.63%, AUC 99.64%
- **Tuberculosis**: Precision 99.80%, Recall 100.00%, F1 99.90%, AUC 100.00%
- **COVID-19**: Precision 99.60%, Recall 98.40%, F1 98.99%, AUC 99.95%
- **Pneumonia**: Precision 95.79%, Recall 95.60%, F1 95.70%, AUC 99.71%

### 4.7 Full MEDXPERT Reference (`EXP-010`)
- **Normal**: Precision 94.82%, Recall 95.20%, F1 95.01%, AUC 99.64%
- **Tuberculosis**: Precision 100.00%, Recall 100.00%, F1 100.00%, AUC 100.00%
- **COVID-19**: Precision 98.41%, Recall 99.20%, F1 98.80%, AUC 99.96%
- **Pneumonia**: Precision 95.34%, Recall 94.20%, F1 94.77%, AUC 99.64%

---

## 5. Early Stopping and Configuration Notes

1. **ResNet-50 (`EXP-001`)**: Completed all 30 epochs. Best validation macro F1 (`0.9763`) achieved at epoch 27.
2. **DenseNet-121 (`EXP-002`)**: Early-stopped at epoch 29 due to 7 consecutive epochs without validation improvement. Best validation macro F1 (`0.9837`) achieved at epoch 22.
3. **EfficientNet-B3 (`EXP-003`)**: Early-stopped at epoch 20 due to 7 consecutive epochs without validation improvement. Best validation macro F1 (`0.9799`) achieved at epoch 13. Image resolution unified to 224 × 224 to ensure exact training budget parity with EXP-010.
4. **ViT-B/16 (`EXP-004`)**: Early-stopped at epoch 21 due to 7 consecutive epochs without validation improvement. Best validation macro F1 (`0.9688`) achieved at epoch 14. Pretrained backbone identifier was updated to `vit_base_patch16_224.orig_in21k_ft_in1k` for compatibility with `timm >= 1.0.0`. Standard learning rate of 3e-4 was maintained.
5. **Swin-T (`EXP-005`)**: Early-stopped at epoch 29 due to 7 consecutive epochs without validation improvement. Best validation macro F1 (`0.9725`) achieved at epoch 22. Learning rate override was removed to inherit the 3e-4 reference budget.
6. **Base ConvNeXt-Tiny (`EXP-A01`)**: Completed all 30 epochs. Best validation macro F1 (`0.9764`) achieved at epoch 30.

---

## 6. Artifact Archive for Paper Publication
All evaluation artifacts (`metrics.json`, `config.yaml`, `history.json`, `confusion_matrix.csv`) for the completed models have been packaged without heavy weight checkpoints (`.pt` excluded):
- **Directory**: `results_for_paper/` (`EXP-001`, `EXP-002`, `EXP-003`, `EXP-004`, `EXP-005`, `EXP-A01`, `EXP-010`)
- **Archive File**: `results_for_paper.zip` (32 KB)
