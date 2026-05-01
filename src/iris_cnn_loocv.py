"""
Deep Learning Lab - Part 1: CNN on IRIS Dataset
Author: [Your Name]
Date: February 2026
Description: Implementation of CNN with LOOCV for IRIS classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import time
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("IRIS Dataset Classification with CNN and LOOCV")
print("=" * 70)
print(f"Python Version: {tf.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("=" * 70)


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_iris_data(filepath):
    """Load and preprocess IRIS dataset"""
    print("\n[1] Loading IRIS Dataset...")

    # Load data
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(filepath, names=column_names)

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nStatistical summary:")
    print(df.describe())

    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    print(f"\nClasses: {label_encoder.classes_}")
    print(f"Encoded labels: {np.unique(y_encoded)}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for CNN (samples, height, width, channels)
    # For IRIS: we'll treat each sample as 2x2 image with 1 channel
    X_reshaped = X_scaled.reshape(-1, 2, 2, 1)

    return X_reshaped, y_categorical, y_encoded, label_encoder, scaler


# ============================================================================
# 2. CNN MODEL ARCHITECTURES
# ============================================================================

def create_cnn_model_v1(input_shape, num_classes):
    """
    CNN Model Version 1: Simple architecture
    - 1 Conv layer
    - 1 MaxPooling
    - Dense layers
    """
    model = models.Sequential([
        layers.Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_cnn_model_v2(input_shape, num_classes):
    """
    CNN Model Version 2: Deeper architecture
    - 2 Conv layers
    - 2 MaxPooling
    - More dense layers
    """
    model = models.Sequential([
        layers.Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (1, 1), activation='relu', padding='same'),
        layers.MaxPooling2D((1, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_cnn_model_v3(input_shape, num_classes):
    """
    CNN Model Version 3: Without MaxPooling
    - 2 Conv layers
    - No MaxPooling (to test the effect)
    - Dense layers
    """
    model = models.Sequential([
        layers.Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_cnn_model_v4(input_shape, num_classes):
    """
    CNN Model Version 4: Different activation function (tanh)
    """
    model = models.Sequential([
        layers.Conv2D(32, (2, 2), activation='tanh', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (1, 1), activation='tanh', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_cnn_model_v5(input_shape, num_classes):
    """
    CNN Model Version 5: Larger filters and different stride
    """
    model = models.Sequential([
        layers.Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (1, 1), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ============================================================================
# 3. LEAVE-ONE-OUT CROSS-VALIDATION
# ============================================================================

def perform_loocv(X, y, y_encoded, model_creator, model_name, epochs=50, verbose=0):
    """
    Perform Leave-One-Out Cross-Validation

    Parameters:
    - X: Input features
    - y: One-hot encoded labels
    - y_encoded: Integer encoded labels
    - model_creator: Function to create the model
    - model_name: Name of the model
    - epochs: Number of training epochs
    - verbose: Verbosity level
    """
    print(f"\n{'=' * 70}")
    print(f"[LOOCV] Testing {model_name}")
    print(f"{'=' * 70}")

    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(X)

    predictions = []
    true_labels = []
    accuracies = []

    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(loo.split(X), 1):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and compile model
        model = model_creator(X.shape[1:], y.shape[1])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )

        # Predict
        y_pred = model.predict(X_test, verbose=0)
        pred_class = np.argmax(y_pred, axis=1)[0]
        true_class = y_encoded[test_index][0]

        predictions.append(pred_class)
        true_labels.append(true_class)

        # Calculate accuracy for this fold
        acc = 1 if pred_class == true_class else 0
        accuracies.append(acc)

        if fold % 10 == 0 or fold == n_splits:
            print(f"Fold {fold}/{n_splits} completed... Current accuracy: {np.mean(accuracies):.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate final metrics
    final_accuracy = accuracy_score(true_labels, predictions)

    print(f"\n{model_name} Results:")
    print(f"Total accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"Correct predictions: {sum(accuracies)}/{n_splits}")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per fold: {total_time / n_splits:.2f} seconds")

    return predictions, true_labels, final_accuracy, total_time


# ============================================================================
# 4. MODEL COMPARISON AND VISUALIZATION
# ============================================================================

def compare_models(results_dict, label_encoder):
    """Compare all models and visualize results"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Create comparison table
    comparison_data = []
    for model_name, result in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Accuracy (%)': f"{result['accuracy'] * 100:.2f}%",
            'Training Time (s)': f"{result['time']:.2f}",
            'Correct/Total': f"{int(result['accuracy'] * len(result['predictions']))}/{len(result['predictions'])}"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy comparison
    model_names = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in model_names]

    axes[0, 0].bar(range(len(model_names)), accuracies, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xlabel('Model', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # 2. Training time comparison
    times = [results_dict[m]['time'] for m in model_names]

    axes[0, 1].bar(range(len(model_names)), times, color='lightcoral', edgecolor='darkred')
    axes[0, 1].set_xlabel('Model', fontsize=12)
    axes[0, 1].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Confusion Matrix for best model
    best_predictions = best_model[1]['predictions']
    best_true_labels = best_model[1]['true_labels']

    cm = confusion_matrix(best_true_labels, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 0].set_ylabel('True Label', fontsize=12)
    axes[1, 0].set_title(f'Confusion Matrix - {best_model[0]}', fontsize=14, fontweight='bold')

    # 4. Accuracy vs Time scatter plot
    axes[1, 1].scatter(times, accuracies, s=200, c='green', alpha=0.6, edgecolors='darkgreen', linewidths=2)

    for i, model in enumerate(model_names):
        axes[1, 1].annotate(model, (times[i], accuracies[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)

    axes[1, 1].set_xlabel('Training Time (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved as 'model_comparison.png'")
    plt.show()

    return best_model


def plot_detailed_confusion_matrices(results_dict, label_encoder):
    """Plot confusion matrices for all models"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (model_name, result) in enumerate(results_dict.items()):
        cm = confusion_matrix(result['true_labels'], result['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[idx],
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    cbar_kws={'label': 'Count'})

        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}',
                            fontsize=12, fontweight='bold')

    # Hide extra subplot
    if n_models < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ All confusion matrices saved as 'all_confusion_matrices.png'")
    plt.show()


def generate_classification_reports(results_dict, label_encoder):
    """Generate detailed classification reports for all models"""
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 70)

    for model_name, result in results_dict.items():
        print(f"\n{'─' * 70}")
        print(f"{model_name}")
        print(f"{'─' * 70}")
        print(classification_report(
            result['true_labels'],
            result['predictions'],
            target_names=label_encoder.classes_,
            digits=4
        ))


# ============================================================================
# 5. FEATURE ANALYSIS
# ============================================================================

def analyze_features(X_original, y_encoded, label_encoder):
    """Analyze and visualize features"""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    # Create feature analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, feature_name in enumerate(feature_names):
        ax = axes[idx // 2, idx % 2]

        for class_idx, class_name in enumerate(label_encoder.classes_):
            mask = y_encoded == class_idx
            ax.hist(X_original[mask, idx], alpha=0.6, label=class_name, bins=20)

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {feature_name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Feature distributions saved as 'feature_distributions.png'")
    plt.show()

    # Correlation matrix
    df_features = pd.DataFrame(X_original, columns=feature_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Feature correlation matrix saved as 'feature_correlation.png'")
    plt.show()


# ============================================================================
# 6. TRAIN FINAL MODEL WITH HISTORY
# ============================================================================

def train_final_model(X, y, model_creator, model_name, epochs=100):
    """Train final model and plot training history"""
    print(f"\n{'=' * 70}")
    print(f"Training Final Model: {model_name}")
    print(f"{'=' * 70}")

    # Split data for training and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    model = model_creator(X.shape[1:], y.shape[1])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nModel Architecture:")
    model.summary()

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history saved as 'training_history.png'")
    plt.show()

    # Save model
    model.save(f'{model_name.replace(" ", "_")}_final.h5')
    print(f"✓ Model saved as '{model_name.replace(' ', '_')}_final.h5'")

    return model, history


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Load data
    X, y, y_encoded, label_encoder, scaler = load_iris_data('../dataset/iris_clean.data')

    # Store original features for analysis
    df = pd.read_csv('../dataset/iris_clean.data',
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    X_original = df.iloc[:, :-1].values

    # Analyze features
    analyze_features(X_original, y_encoded, label_encoder)

    # Define models to test
    models_to_test = {
        'CNN v1 (Simple)': create_cnn_model_v1,
        'CNN v2 (Deep)': create_cnn_model_v2,
        'CNN v3 (No MaxPool)': create_cnn_model_v3,
        'CNN v4 (Tanh)': create_cnn_model_v4,
        'CNN v5 (Large Filters)': create_cnn_model_v5
    }

    # Perform LOOCV for all models
    results_dict = {}

    for model_name, model_creator in models_to_test.items():
        predictions, true_labels, accuracy, training_time = perform_loocv(
            X, y, y_encoded, model_creator, model_name, epochs=30, verbose=0
        )

        results_dict[model_name] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'time': training_time
        }

    # Compare all models
    best_model_info = compare_models(results_dict, label_encoder)

    # Plot all confusion matrices
    plot_detailed_confusion_matrices(results_dict, label_encoder)

    # Generate classification reports
    generate_classification_reports(results_dict, label_encoder)

    # Train final model with best architecture
    best_model_name = best_model_info[0]
    best_model_creator = models_to_test[best_model_name]

    final_model, history = train_final_model(
        X, y, best_model_creator, best_model_name, epochs=100
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  ✓ model_comparison.png")
    print("  ✓ all_confusion_matrices.png")
    print("  ✓ feature_distributions.png")
    print("  ✓ feature_correlation.png")
    print("  ✓ training_history.png")
    print(f"  ✓ {best_model_name.replace(' ', '_')}_final.h5")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()