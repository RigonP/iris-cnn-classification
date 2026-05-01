import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                            accuracy_score, precision_score, recall_score,
                            f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from itertools import combinations
import warnings
import os
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

print("="*80)
print("Leave-One-Out Cross-Validation on IRIS Dataset")
print("="*80)

# LOADING AND EXPLORING THE DATA
print("\n" + "="*80)
print("1. LOADING AND EXPLORING THE IRIS DATASET")
print("="*80)

# Load IRIS dataset from file
column_names = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)', 'class']

# Read the clean CSV file (no missing values - matches sklearn's iris dataset)
df = pd.read_csv('../dataset/iris_clean.data', header=None, names=column_names)


# Map class names to numeric values
class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
df['target'] = df['class'].map(class_mapping)

# Extract features and target
X = df[column_names[:-1]].values
y = df['target'].values
feature_names = column_names[:-1]
target_names = ['setosa', 'versicolor', 'virginica']

print(f"\nDataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target classes: {target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Target_name column for display
df['target_name'] = df['target'].map({0: target_names[0],
                                       1: target_names[1],
                                       2: target_names[2]})

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())


# SIMPLE SCATTER PLOT (PETAL LENGTH vs PETAL WIDTH)
print("\n" + "="*80)
print("1.5. PETAL SCATTER PLOT VISUALIZATION")
print("="*80)

plt.figure(figsize=(10, 8))
for class_id in np.unique(y):
    plt.scatter(
        X[y == class_id, 2],  # Petal Length
        X[y == class_id, 3],  # Petal Width
        label=target_names[class_id],
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
plt.xlabel("Petal Length (cm)", fontsize=12)
plt.ylabel("Petal Width (cm)", fontsize=12)
plt.legend(fontsize=11)
plt.title("Scatter Plot of IRIS Dataset - Petal Features", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/iris_petal_scatter.png', dpi=300, bbox_inches='tight')
print("\n✓ Petal scatter plot saved as '../results/iris_petal_scatter.png'")
plt.close()

