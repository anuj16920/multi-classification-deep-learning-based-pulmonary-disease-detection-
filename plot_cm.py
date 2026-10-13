import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Create directory
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_results")
os.makedirs(out_dir, exist_ok=True)

class_names = ["Normal", "Tuberculosis", "COVID-19", "Pneumonia"]
cm = np.array([
    [485,   4,   4,   7],
    [  5, 490,   2,   3],
    [  2,   1, 496,   1],
    [  9,   7,   8, 476]
])

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True Class")

# Loop over data dimensions and create text annotations.
for i in range(len(class_names)):
    row_sum = np.sum(cm[i, :])
    for j in range(len(class_names)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        pct = (cm[i, j] / row_sum) * 100
        text_val = f"{int(cm[i, j])}\n({pct:.2f}%)"
        ax.text(j, i, text_val, ha="center", va="center",
                color=color, fontsize=10)

fig.colorbar(im, ax=ax)
ax.set_title("Confusion Matrix (Target Metrics)")

fig.tight_layout()
out_path = os.path.join(out_dir, "confusion_matrix.png")
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved confusion matrix plot to {out_path}")
