
import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Separate features and target
X = iris.data
y = iris.target

# Dataset information
print("IRIS FLOWER CLASSIFICATION")
print("--------------------------")

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature Names:")
print(iris.feature_names)

print("\nFlower Categories:")
print(iris.target_names)

# Data type
print("\nData Type:")
print(X.dtype)

# Missing values
print("\nMissing Values:")
print(np.isnan(X).sum())

# First five rows
print("\nFirst Five Rows:")
print(X[:5])

# Summary statistics
print("\nMean Values:")
print(np.round(np.mean(X, axis=0), 2))

print("\nMinimum Values:")
print(np.min(X, axis=0))

print("\nMaximum Values:")
print(np.max(X, axis=0))

# Flower category counts
print("\nFlower Category Counts:")
print(np.bincount(y))

print("\nDataset Loaded Successfully!")
