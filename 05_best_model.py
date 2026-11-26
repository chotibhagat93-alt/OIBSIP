
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train best model
best_model = LogisticRegression(max_iter=200)
best_model.fit(X_train, y_train)

# Predictions
predictions = best_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("BEST MODEL: LOGISTIC REGRESSION")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))

# Save model
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

joblib.dump(best_model, output_dir / "best_iris_model.pkl")

print("\nBest model saved successfully!")
