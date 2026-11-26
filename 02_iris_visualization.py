
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 1. Scatter Matrix
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for row in range(4):
    for col in range(4):
        ax = axes[row, col]

        if row == col:
            for class_index, species in enumerate(target_names):
                values = X[y == class_index, row]
                ax.hist(values, bins=12, alpha=0.55, label=species)
        else:
            for class_index, species in enumerate(target_names):
                mask = y == class_index
                ax.scatter(
                    X[mask, col],
                    X[mask, row],
                    alpha=0.65,
                    s=18,
                    label=species
                )

        if row == 3:
            ax.set_xlabel(feature_names[col])

        if col == 0:
            ax.set_ylabel(feature_names[row])

fig.suptitle("Iris Flower Scatter Matrix")
fig.tight_layout()

scatter_path = output_dir / "iris_scatter_matrix.png"
fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Boxplots
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for feature_index, ax in enumerate(axes):
    grouped_data = [
        X[y == class_index, feature_index]
        for class_index in range(len(target_names))
    ]

    ax.boxplot(grouped_data)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(target_names, rotation=25)
    ax.set_title(feature_names[feature_index])
    ax.set_ylabel("Measurement (cm)")

fig.suptitle("Iris Flower Boxplots")
fig.tight_layout()

boxplot_path = output_dir / "iris_boxplots.png"
fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Iris Visualization Completed!")
print("Dataset shape:", X.shape)
print("Scatter Matrix saved:", scatter_path)
print("Boxplots saved:", boxplot_path)
print("\nBoth graphs generated successfully!")
