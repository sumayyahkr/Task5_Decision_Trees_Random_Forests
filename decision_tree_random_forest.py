# Import libraries
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# Explore dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["target"].value_counts())

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

# Predictions
tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

# Evaluate accuracy
print("\nDECISION TREE RESULTS")

print("Training Accuracy:",
      accuracy_score(y_train, tree_train_pred))

print("Testing Accuracy:",
      accuracy_score(y_test, tree_test_pred))

# Decision Tree with limited depth
small_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

small_tree.fit(X_train, y_train)

small_train_pred = small_tree.predict(X_train)
small_test_pred = small_tree.predict(X_test)

print("\nLIMITED DEPTH DECISION TREE")

print("Training Accuracy:",
      accuracy_score(y_train, small_train_pred))

print("Testing Accuracy:",
      accuracy_score(y_test, small_test_pred))
from sklearn.ensemble import RandomForestClassifier
# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Accuracy
print("\nRANDOM FOREST RESULTS")

print("Training Accuracy:",
      accuracy_score(y_train, rf_train_pred))

print("Testing Accuracy:",
      accuracy_score(y_test, rf_test_pred))
from sklearn.model_selection import cross_val_score
# Cross-validation
tree_scores = cross_val_score(
    small_tree, X, y, cv=5, scoring="accuracy"
)

rf_scores = cross_val_score(
    rf_model, X, y, cv=5, scoring="accuracy"
)

print("\nCROSS VALIDATION RESULTS")

print("Decision Tree CV scores:", tree_scores)
print("Decision Tree Mean:", tree_scores.mean())

print("\nRandom Forest CV scores:", rf_scores)
print("Random Forest Mean:", rf_scores.mean())

import matplotlib.pyplot as plt

# Feature importance
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"].head(10),
    importance_df["Importance"].head(10)
)

plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()
from sklearn.tree import plot_tree
# Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    small_tree,
    feature_names=X.columns,
    class_names=["Malignant", "Benign"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()