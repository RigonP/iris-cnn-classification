"""
This script implements a comprehensive CNN-based classification system for the IRIS dataset
with Leave-One-Out Cross-Validation (LOOCV) and extensive hyperparameter experimentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
from data_loader import load_iris_arrays
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IrisDataProcessor:
    """Class to handle IRIS dataset loading and preprocessing"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_and_prepare_data(self):
        """Load IRIS dataset from dataset/iris.data (Moodle source).

        Missing values in the raw file are imputed with the column-wise
        median by data_loader.load_iris_arrays.
        """
        X, y, feature_names, target_names = load_iris_arrays()

        print("=" * 70)
        print("IRIS Dataset Information")
        print("=" * 70)
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Feature names: {feature_names}")
        print(f"Target classes: {target_names}")
        print(f"Class distribution: {np.bincount(y)}")
        print("=" * 70)

        return X, y, feature_names, target_names

    def preprocess_data(self, X, y):
        """Standardize features and encode labels"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # For CNN, we need to reshape data to have a channel dimension
        # Shape: (samples, features, 1) to mimic image-like data
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # One-hot encode labels
        y_encoded = to_categorical(y, num_classes=3)

        return X_reshaped, y_encoded, y


class CNNModelBuilder:
    """Class to build different CNN architectures"""

    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_baseline_cnn(self):
        """Build a baseline CNN model"""
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=2, activation='relu',
                         input_shape=self.input_shape, padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def build_deep_cnn(self):
        """Build a deeper CNN with more layers"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=2, activation='relu',
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def build_cnn_without_pooling(self):
        """Build CNN without max-pooling layers"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=2, activation='relu',
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def build_cnn_different_activations(self, activation='tanh'):
        """Build CNN with different activation functions"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=2, activation=activation,
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=2, activation=activation, padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation=activation),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def build_cnn_l2_regularization(self):
        """Build CNN with L2 regularization"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=2, activation='relu',
                         input_shape=self.input_shape, padding='same',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model


class LOOCVEvaluator:
    """Class to perform Leave-One-Out Cross-Validation"""

    def __init__(self, model_builder_func, input_shape):
        self.model_builder_func = model_builder_func
        self.input_shape = input_shape

    def perform_loocv(self, X, y, epochs=50, verbose=0):
        """
        Perform Leave-One-Out Cross-Validation

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features, 1)
            Training data
        y : array-like, shape (n_samples, n_classes)
            Target labels (one-hot encoded)
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level

        Returns:
        --------
        predictions : list
            Predictions for each left-out sample
        true_labels : list
            True labels for each left-out sample
        """
        loo = LeaveOneOut()
        predictions = []
        true_labels = []

        print(f"\nPerforming LOOCV with {X.shape[0]} iterations...")

        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            # Split data
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Build fresh model
            model = self.model_builder_func()

            # Train model
            early_stop = EarlyStopping(monitor='loss', patience=10,
                                      restore_best_weights=True)

            model.fit(X_train_fold, y_train_fold,
                     epochs=epochs,
                     verbose=verbose,
                     callbacks=[early_stop])

            # Predict on the left-out sample
            pred = model.predict(X_test_fold, verbose=0)
            predictions.append(np.argmax(pred, axis=1)[0])
            true_labels.append(np.argmax(y_test_fold, axis=1)[0])

            if (fold + 1) % 30 == 0:
                print(f"Completed {fold + 1}/{X.shape[0]} folds")

        return predictions, true_labels


class ModelEvaluator:
    """Class to evaluate model performance"""

    @staticmethod
    def calculate_metrics(y_true, y_pred, target_names):
        """Calculate and display various metrics"""
        accuracy = accuracy_score(y_true, y_pred)

        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        print(f"Error Rate: {(1 - accuracy) * 100:.2f}%")
        print("\nClassification Report:")
        print("-" * 70)
        print(classification_report(y_true, y_pred, target_names=target_names))

        return accuracy

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, target_names, model_name, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm


class TrainingVisualizer:
    """Class to visualize training process"""

    @staticmethod
    def plot_training_history(history, model_name, save_path=None):
        """Plot training and validation accuracy/loss"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy',
                    linewidth=2, marker='o', markersize=4)
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                        linewidth=2, marker='s', markersize=4)
        axes[0].set_title(f'Model Accuracy - {model_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss',
                    linewidth=2, marker='o', markersize=4, color='red')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss',
                        linewidth=2, marker='s', markersize=4, color='orange')
        axes[1].set_title(f'Model Loss - {model_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FeatureAnalyzer:
    """Class to analyze learned features"""

    @staticmethod
    def analyze_feature_importance(model, X, feature_names):
        """
        Analyze feature importance using gradient-based method
        This helps identify which features the CNN has learned
        """
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

        # Get predictions
        predictions = model.predict(X, verbose=0)

        # Calculate feature importance using gradients
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)
            # Use mean prediction as target
            target = tf.reduce_mean(predictions)

        # Get gradients
        gradients = tape.gradient(target, X_tensor)

        # Calculate importance scores (mean absolute gradient per feature)
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=(0, 2))

        # Normalize scores
        importance_scores = importance_scores / np.sum(importance_scores)

        # Display results
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance Score': importance_scores
        }).sort_values('Importance Score', ascending=False)

        print("\nFeature Importance Ranking:")
        print("-" * 70)
        print(feature_importance_df.to_string(index=False))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance_scores, color='steelblue', alpha=0.8)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('CNN Learned Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        # Create results directory if it doesn't exist
        os.makedirs('../results', exist_ok=True)
        plt.savefig(
            os.path.join(RESULTS_DIR, "feature_importance.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()

        return feature_importance_df


def compare_model_configurations(X, y, y_original, input_shape, target_names):
    """
    Compare different CNN configurations
    """
    print("\n" + "=" * 70)
    print("COMPARING DIFFERENT CNN CONFIGURATIONS")
    print("=" * 70)

    # Split data for regular training (not LOOCV for faster comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_original
    )

    # Also split original labels
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y_original, test_size=0.2, random_state=42, stratify=y_original
    )

    model_builder = CNNModelBuilder(input_shape)

    # Define model configurations
    configurations = {
        'Baseline CNN': model_builder.build_baseline_cnn,
        'Deep CNN': model_builder.build_deep_cnn,
        'CNN without MaxPooling': model_builder.build_cnn_without_pooling,
        'CNN with Tanh Activation': lambda: model_builder.build_cnn_different_activations('tanh'),
        'CNN with L2 Regularization': model_builder.build_cnn_l2_regularization
    }

    results = {}

    for config_name, build_func in configurations.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {config_name}")
        print(f"{'=' * 70}")

        # Build model
        model = build_func()

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        # Setup callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                  restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=5, verbose=1, min_lr=1e-7)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Evaluate model
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy = accuracy_score(y_test_orig, y_pred)

        # Store results
        results[config_name] = {
            'model': model,
            'history': history,
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test_orig
        }

        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

        # Plot training history
        TrainingVisualizer.plot_training_history(
            history, config_name,
            save_path=os.path.join(
                RESULTS_DIR,
                f"training_history_{config_name.replace(' ', '_')}.png"
            )

        )

        # Plot confusion matrix
        ModelEvaluator.plot_confusion_matrix(
            y_test_orig, y_pred, target_names, config_name,
            save_path=os.path.join(
                RESULTS_DIR,
                f"confusion_matrix_{config_name.replace(' ', '_')}.png"
            )

        )

    return results


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("DEEP LEARNING LAB PROJECT - PART 1: IRIS CLASSIFICATION")
    print("=" * 70)

    # 1. Load and prepare data
    print("\n[Step 1] Loading and preparing IRIS dataset...")
    data_processor = IrisDataProcessor()
    X, y, feature_names, target_names = data_processor.load_and_prepare_data()
    X_processed, y_encoded, y_original = data_processor.preprocess_data(X, y)

    input_shape = (X_processed.shape[1], 1)
    print(f"Input shape for CNN: {input_shape}")

    # 2. Compare different model configurations
    print("\n[Step 2] Comparing different CNN configurations...")
    results = compare_model_configurations(
        X_processed, y_encoded, y_original, input_shape, target_names
    )

    # 3. Print comparison summary
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Accuracy':>15}")
    print("-" * 70)
    for config_name, result in results.items():
        print(f"{config_name:<35} {result['accuracy']*100:>14.2f}%")

    # 4. Select best model and perform LOOCV
    best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_config_name = best_config[0]
    print(f"\nBest Configuration: {best_config_name}")
    print(f"Accuracy: {best_config[1]['accuracy']*100:.2f}%")

    # 5. Perform LOOCV with best model
    print("\n[Step 3] Performing Leave-One-Out Cross-Validation with best model...")
    model_builder = CNNModelBuilder(input_shape)

    # Map configuration name to builder function
    config_to_builder = {
        'Baseline CNN': model_builder.build_baseline_cnn,
        'Deep CNN': model_builder.build_deep_cnn,
        'CNN without MaxPooling': model_builder.build_cnn_without_pooling,
        'CNN with Tanh Activation': lambda: model_builder.build_cnn_different_activations('tanh'),
        'CNN with L2 Regularization': model_builder.build_cnn_l2_regularization
    }

    loocv_evaluator = LOOCVEvaluator(
        config_to_builder[best_config_name],
        input_shape
    )

    loocv_predictions, loocv_true = loocv_evaluator.perform_loocv(
        X_processed, y_encoded, epochs=50, verbose=0
    )

    # 6. Evaluate LOOCV results
    print("\n[Step 4] Evaluating LOOCV results...")
    loocv_accuracy = ModelEvaluator.calculate_metrics(
        loocv_true, loocv_predictions, target_names
    )

    ModelEvaluator.plot_confusion_matrix(
        loocv_true, loocv_predictions, target_names,
        f"{best_config_name} - LOOCV",
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix_loocv.png")
    )

    # 7. Feature importance analysis
    print("\n[Step 5] Analyzing learned features...")
    best_model = best_config[1]['model']
    feature_importance = FeatureAnalyzer.analyze_feature_importance(
        best_model, X_processed, feature_names
    )

    # 8. Generate summary report
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Best Model Configuration: {best_config_name}")
    print(f"Test Set Accuracy: {best_config[1]['accuracy']*100:.2f}%")
    print(f"LOOCV Accuracy: {loocv_accuracy*100:.2f}%")
    print(f"\nMost Important Features (learned by CNN):")
    print(feature_importance.head(4).to_string(index=False))
    print("\n" + "=" * 70)
    print("All results and visualizations have been saved!")
    print("=" * 70)

    return results, feature_importance


if __name__ == "__main__":
    # Print system information
    print("\nSystem Information:")
    print(f"Python Version: {tf.__version__}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Available: {len(gpus)} device(s)")
    else:
        print("GPU: Not available (using CPU)")

    # Run main analysis
    results, feature_importance = main()

    print("\n✓ Analysis completed successfully!")