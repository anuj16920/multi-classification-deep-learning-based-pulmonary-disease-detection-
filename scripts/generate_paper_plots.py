#!/usr/bin/env python3
"""
Generate publication-quality training curves and confusion matrix for MEDXPERT (EXP-010).
Produces:
  - outputs/EXP-010/learning_curves.png (4-panel high-res publication figure)
  - outputs/EXP-010/paper_confusion_matrix.png (High-res formatted CM with counts & percentages)
  - outputs/EXP-010/ablation_comparison.png (Ablation study visual scorecard)
  - outputs/EXP-010/baseline_comparison.png (Baseline models benchmark comparison)
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Global publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

EXP_DIR = Path("outputs/EXP-010")
CLASS_NAMES = ["Normal", "Tuberculosis", "COVID-19", "Pneumonia"]

def generate_learning_curves():
    history_file = EXP_DIR / "history.json"
    with open(history_file) as f:
        history = json.load(f)

    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]
    val_acc = [x["accuracy"] * 100 for x in history]
    val_f1 = [x["f1_macro"] * 100 for x in history]
    val_auc = [x["auc_ovr_macro"] * 100 for x in history]

    # Best epoch
    best_idx = int(np.argmax(val_f1))
    best_epoch = epochs[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MEDXPERT Training & Validation Dynamics (EXP-010, 30 Epochs)", fontsize=16, fontweight='bold', y=0.98)

    # Panel A: Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'o-', color='#1f77b4', lw=2.2, label='Train Loss (CE + Entropy)', markersize=4)
    ax.plot(epochs, val_loss, 's-', color='#d62728', lw=2.2, label='Validation Loss', markersize=4)
    ax.axvline(best_epoch, color='green', linestyle=':', lw=1.8, label=f'Best Checkpoint (Ep {best_epoch})')
    ax.set_title('(a) Loss Convergence Curve', fontweight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(-0.05, max(val_loss) * 1.15)

    # Panel B: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, val_acc, 'o-', color='#2ca02c', lw=2.2, label='Validation Accuracy (%)', markersize=4)
    ax.axhline(97.15, color='#9467bd', linestyle='--', lw=1.8, label='Test Accuracy (97.15%)')
    ax.axhline(97.35, color='#8c564b', linestyle=':', lw=1.8, label='Paper Benchmark (97.35%)')
    ax.scatter([best_epoch], [val_acc[best_idx]], color='red', s=80, zorder=5, label=f'Peak Val: {val_acc[best_idx]:.2f}%')
    ax.set_title('(b) Classification Accuracy Progression', fontweight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(80, 100)

    # Panel C: F1 Score
    ax = axes[1, 0]
    ax.plot(epochs, val_f1, 'd-', color='#ff7f0e', lw=2.2, label='Macro F1-Score (%)', markersize=4)
    ax.axhline(97.15, color='#17becf', linestyle='--', lw=1.8, label='Test F1 (97.15%)')
    ax.scatter([best_epoch], [val_f1[best_idx]], color='red', s=80, zorder=5, label=f'Peak F1: {val_f1[best_idx]:.2f}%')
    ax.set_title('(c) Macro F1-Score Progression', fontweight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Macro F1 (%)')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(80, 100)

    # Panel D: ROC-AUC
    ax = axes[1, 1]
    ax.plot(epochs, val_auc, '^-', color='#9467bd', lw=2.2, label='Macro One-vs-Rest AUC (%)', markersize=4)
    ax.axhline(99.81, color='#e377c2', linestyle='--', lw=1.8, label='Test AUC (99.81%)')
    ax.axhline(99.89, color='#7f7f7f', linestyle=':', lw=1.8, label='Paper Benchmark (99.89%)')
    ax.set_title('(d) Discriminative Power (Macro ROC-AUC)', fontweight='semibold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC (%)')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(96, 100.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = EXP_DIR / "learning_curves.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved learning curves to {out_file}")


def generate_paper_confusion_matrix():
    # Read actual predictions from prediction_results.csv
    import pandas as pd
    preds_file = EXP_DIR / "prediction_results.csv"
    df = pd.read_csv(preds_file)
    y_true = df["label"].values
    y_pred = df["pred"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, ax=ax, fmt='d',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

    # Overlay custom dual-text: Count and percentage
    for i in range(4):
        for j in range(4):
            count = cm[i, j]
            pct = cm_percent[i, j]
            text_color = "white" if count > 250 else "black"
            ax.text(j + 0.5, i + 0.4, f"{count}", ha="center", va="center",
                    color=text_color, fontweight="bold", fontsize=13)
            ax.text(j + 0.5, i + 0.65, f"({pct:.1f}%)", ha="center", va="center",
                    color=text_color, fontsize=10)

    ax.set_title("MEDXPERT Official Test Confusion Matrix (N=2,000)\nOverall Accuracy: 97.15% | Macro F1: 97.15%",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontweight='semibold', labelpad=8)
    ax.set_ylabel("True Ground Truth", fontweight='semibold', labelpad=8)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)

    out_file = EXP_DIR / "paper_confusion_matrix.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved paper confusion matrix to {out_file}")


def generate_comparison_charts():
    # 1. Baseline Models Comparison
    baselines = [
        ("ResNet50", 92.84, 93.10, 92.84, 92.95, 98.42),
        ("DenseNet121", 94.67, 94.80, 94.67, 94.72, 99.12),
        ("Vision Transformer", 94.92, 95.10, 94.92, 95.00, 99.25),
        ("EfficientNet-B3", 95.51, 95.60, 95.51, 95.55, 99.45),
        ("Swin Transformer", 96.08, 96.15, 96.08, 96.11, 99.62),
        ("MEDXPERT (Proposed)", 97.15, 97.14, 97.15, 97.15, 99.81),
    ]

    names = [b[0] for b in baselines]
    accs = [b[1] for b in baselines]
    f1s = [b[4] for b in baselines]
    aucs = [b[5] for b in baselines]

    x = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width, accs, width, label='Accuracy (%)', color='#4a90e2')
    r2 = ax.bar(x, f1s, width, label='F1-Score (%)', color='#50e3c2')
    r3 = ax.bar(x + width, aucs, width, label='ROC-AUC (%)', color='#b8e986')

    ax.set_ylabel('Percentage (%)', fontweight='semibold')
    ax.set_title('Comparative Benchmark: Proposed MEDXPERT vs Baseline SOTA Models', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontweight='semibold')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(90, 101)

    # Highlight MEDXPERT
    ax.get_xticklabels()[-1].set_color('#d0021b')
    ax.get_xticklabels()[-1].set_fontweight('bold')

    for r in [r1, r2, r3]:
        for bar in r:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    out_file = EXP_DIR / "baseline_comparison.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved baseline comparison chart to {out_file}")

    # 2. Ablation Study Chart
    ablations = [
        ("Backbone Only", 87.87, 88.25, 87.90, 97.71),
        ("+ CBAM Attention", 85.06, 85.38, 85.13, 96.78),
        ("+ Adaptive DyDA Router", 86.52, 87.09, 86.51, 97.65),
        ("+ Swin Hierarchical Feature", 95.51, 95.54, 95.48, 99.69),
        ("+ Full Graph & Memory (MEDXPERT)", 97.15, 97.14, 97.15, 99.81),
    ]

    a_names = [a[0] for a in ablations]
    a_accs = [a[1] for a in ablations]
    a_f1s = [a[3] for a in ablations]
    a_aucs = [a[4] for a in ablations]

    xa = np.arange(len(a_names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(xa, a_accs, 'o-', color='#e67e22', lw=2.5, markersize=8, label='Accuracy (%)')
    ax.plot(xa, a_f1s, 's--', color='#2980b9', lw=2.5, markersize=8, label='F1-Score (%)')
    ax.plot(xa, a_aucs, '^-.', color='#27ae60', lw=2.5, markersize=8, label='ROC-AUC (%)')

    for i in range(len(a_names)):
        ax.annotate(f"{a_accs[i]:.2f}%", (xa[i], a_accs[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#e67e22')

    ax.set_xticks(xa)
    ax.set_xticklabels(a_names, rotation=15, ha='right', fontweight='semibold')
    ax.set_ylabel('Percentage (%)', fontweight='semibold')
    ax.set_title('Ablation Study: Progressive Architectural Contribution to Diagnostic Performance', fontweight='bold', fontsize=13)
    ax.set_ylim(83, 102)
    ax.legend(loc='lower right', frameon=True)

    out_file = EXP_DIR / "ablation_comparison.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved ablation comparison chart to {out_file}")

if __name__ == "__main__":
    generate_learning_curves()
    generate_paper_confusion_matrix()
    generate_comparison_charts()
