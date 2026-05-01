"""
Advanced Analysis and Visualization Module
This script provides additional analysis tools for the IRIS CNN project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import models
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create results directory
os.makedirs('../results', exist_ok=True)


class AdvancedVisualizer:
    """Advanced visualization tools for deep learning analysis"""

    @staticmethod
    def plot_dataset_distribution(X, y, feature_names, target_names):
        """
        Visualize the distribution of IRIS dataset features
        """
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = [target_names[i] for i in y]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IRIS Dataset Feature Distributions',
                    fontsize=16, fontweight='bold', y=1.02)

        # Plot each feature
        for idx, feature in enumerate(feature_names):
            ax = axes[idx // 2, idx % 2]

            for species in target_names:
                data = df[df['species'] == species][feature]
                ax.hist(data, alpha=0.6, label=species, bins=20, edgecolor='black')

            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        return df

    @staticmethod
    def plot_correlation_matrix(X, feature_names):
        """
        Plot correlation matrix of features
        """
        # Create correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=feature_names, yticklabels=feature_names,
                   center=0, vmin=-1, vmax=1, square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_pairwise_features(X, y, feature_names, target_names):
        """
        Create pairwise scatter plots of features
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = y

        # Create pair plot
        colors = ['red', 'blue', 'green']

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('Pairwise Feature Relationships',
                    fontsize=16, fontweight='bold', y=0.995)

        for i in range(4):
            for j in range(4):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    for class_idx, class_name in enumerate(target_names):
                        class_data = X[y == class_idx, i]
                        ax.hist(class_data, alpha=0.6, label=class_name,
                               bins=15, color=colors[class_idx])
                    ax.set_ylabel('Frequency')
                    if i == 0:
                        ax.legend(loc='upper right', fontsize=8)
                else:
                    # Off-diagonal: scatter plot
                    for class_idx, class_name in enumerate(target_names):
                        class_mask = y == class_idx
                        ax.scatter(X[class_mask, j], X[class_mask, i],
                                 alpha=0.6, s=30, color=colors[class_idx],
                                 label=class_name if i == 0 and j == 1 else "")

                # Set labels
                if i == 3:
                    ax.set_xlabel(feature_names[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(feature_names[i], fontsize=9)

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/pairwise_features.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_dimensionality_reduction(X, y, target_names):
        """
        Visualize data using PCA and t-SNE
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['red', 'blue', 'green']

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        ax = axes[0]
        for class_idx, class_name in enumerate(target_names):
            class_mask = y == class_idx
            ax.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1],
                      alpha=0.7, s=100, color=colors[class_idx],
                      label=class_name, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA Visualization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)

        ax = axes[1]
        for class_idx, class_name in enumerate(target_names):
            class_mask = y == class_idx
            ax.scatter(X_tsne[class_mask, 0], X_tsne[class_mask, 1],
                      alpha=0.7, s=100, color=colors[class_idx],
                      label=class_name, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.show()


class ModelAnalyzer:
    """Tools for analyzing CNN model internals"""

    @staticmethod
    def visualize_layer_outputs(model, X_sample, layer_names=None):
        """
        Visualize the output of different layers in the CNN
        """
        # Ensure model is built by calling it with a sample
        X_sample = np.atleast_3d(X_sample)
        if X_sample.ndim == 2:
            X_sample = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1])

        # Build the model by making a prediction
        try:
            _ = model.predict(X_sample, verbose=0)
        except:
            pass  # Model might already be built

        if layer_names is None:
            # Get all conv and dense layer names
            layer_names = [layer.name for layer in model.layers
                          if 'conv' in layer.name or 'dense' in layer.name]

        if not layer_names:
            print("No convolutional or dense layers found in model.")
            return

        # Create models for each layer output
        try:
            layer_outputs = [model.get_layer(name).output for name in layer_names]
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        except Exception as e:
            print(f"Could not create activation model: {e}")
            print("Skipping layer activation visualization.")
            return

        # Get activations
        try:
            activations = activation_model.predict(X_sample, verbose=0)

            # Handle single layer case
            if not isinstance(activations, list):
                activations = [activations]
        except Exception as e:
            print(f"Error getting activations: {e}")
            return

        # Plot activations
        n_layers = len(layer_names)
        fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))

        if n_layers == 1:
            axes = [axes]

        fig.suptitle('Layer Activations Visualization',
                    fontsize=14, fontweight='bold')

        for idx, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            ax = axes[idx]

            if len(activation.shape) == 3:  # Conv layer
                # Average across all filters
                avg_activation = np.mean(activation[0], axis=-1)
                ax.plot(avg_activation, linewidth=2, color='steelblue')
                ax.fill_between(range(len(avg_activation)), avg_activation,
                               alpha=0.3, color='steelblue')
            else:  # Dense layer
                ax.bar(range(len(activation[0])), activation[0],
                      color='steelblue', alpha=0.7)

            ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Neuron/Position', fontsize=10)
            ax.set_ylabel('Activation', fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../results/layer_activations.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def analyze_filter_weights(model):
        """
        Analyze and visualize filter weights in convolutional layers
        """
        conv_layers = [layer for layer in model.layers if 'conv' in layer.name]

        if not conv_layers:
            print("No convolutional layers found in model.")
            return

        n_layers = len(conv_layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))

        if n_layers == 1:
            axes = [axes]

        fig.suptitle('Convolutional Filter Weights',
                    fontsize=14, fontweight='bold')

        for idx, layer in enumerate(conv_layers):
            try:
                weights = layer.get_weights()[0]  # Shape: (kernel_size, input_dim, filters)

                # Flatten weights for visualization
                weights_flat = weights.reshape(-1, weights.shape[-1])

                ax = axes[idx]
                im = ax.imshow(weights_flat.T, cmap='viridis', aspect='auto')
                ax.set_title(f'{layer.name}\n({weights.shape[-1]} filters)',
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Weight Position', fontsize=10)
                ax.set_ylabel('Filter Index', fontsize=10)
                plt.colorbar(im, ax=ax, label='Weight Value')
            except Exception as e:
                print(f"Could not visualize weights for layer {layer.name}: {e}")
                ax = axes[idx]
                ax.text(0.5, 0.5, f'Error visualizing\n{layer.name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{layer.name}', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig('../results/filter_weights.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_model_architecture(model, save_path='../results/model_architecture.png'):
        """
        Create a visual representation of model architecture
        """
        try:
            # Try to use plot_model
            tf.keras.utils.plot_model(
                model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=150
            )
            print(f"✓ Model architecture saved to {save_path}")
        except Exception as e:
            print(f"Could not plot model architecture using plot_model: {e}")
            print("Displaying model summary instead:")
            try:
                model.summary()
            except Exception as e2:
                print(f"Could not display model summary: {e2}")
                print("Model details:")
                print(f"  Number of layers: {len(model.layers)}")
                for i, layer in enumerate(model.layers):
                    print(f"  {i+1}. {layer.name} ({layer.__class__.__name__})")


class PerformanceAnalyzer:
    """Tools for analyzing model performance"""

    @staticmethod
    def compare_configurations_detailed(results):
        """
        Create detailed comparison visualizations of different configurations
        """
        config_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in config_names]

        # Create comprehensive comparison plot with maximum spacing
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.4,
                             top=0.96, bottom=0.04, left=0.08, right=0.96)

        # 1. Accuracy comparison bar plot
        ax1 = fig.add_subplot(gs[0, :])
        bars = ax1.bar(range(len(config_names)),
                       [acc * 100 for acc in accuracies],
                       color='steelblue', alpha=0.8, edgecolor='black', width=0.5)
        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels(config_names, rotation=35, ha='right', fontsize=6)
        ax1.set_ylabel('Accuracy (%)', fontsize=7)
        ax1.set_title('Model Configuration Comparison', fontsize=8, fontweight='bold', pad=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([min(accuracies) * 90, 100])
        ax1.tick_params(labelsize=6)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                    f'{acc*100:.1f}%',
                    ha='center', va='bottom', fontsize=5, fontweight='bold')

        # 2-3. Training history for top 2 models
        top_2_configs = sorted(results.items(),
                              key=lambda x: x[1]['accuracy'],
                              reverse=True)[:2]

        for idx, (config_name, result) in enumerate(top_2_configs):
            ax = fig.add_subplot(gs[1, idx])
            history = result['history']

            ax.plot(history.history['accuracy'], label='Train',
                   linewidth=1.2, marker='o', markersize=2)
            if 'val_accuracy' in history.history:
                ax.plot(history.history['val_accuracy'], label='Validation',
                       linewidth=1.2, marker='s', markersize=2)

            ax.set_xlabel('Epoch', fontsize=6)
            ax.set_ylabel('Accuracy', fontsize=6)
            ax.set_title(f'{config_name} (Acc: {result["accuracy"]*100:.2f}%)',
                        fontsize=7, fontweight='bold', pad=8)
            ax.legend(fontsize=5, loc='best', frameon=True, fancybox=False)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=5)

        # 4-5. Confusion matrices for top 2 models
        for idx, (config_name, result) in enumerate(top_2_configs):
            ax = fig.add_subplot(gs[2, idx])

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(result['true_labels'], result['predictions'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count', 'shrink': 0.7},
                       annot_kws={'fontsize': 6})
            ax.set_title(f'{config_name} - Confusion Matrix',
                        fontsize=7, fontweight='bold', pad=8)
            ax.set_ylabel('True', fontsize=6)
            ax.set_xlabel('Predicted', fontsize=6)
            ax.tick_params(labelsize=5)

        plt.suptitle('Comprehensive Model Performance Analysis',
                    fontsize=10, fontweight='bold', y=0.99)
        plt.savefig('../results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def analyze_misclassifications(y_true, y_pred, X, feature_names, target_names):
        """
        Analyze misclassified samples with robust input handling
        """
        # Convert to numpy arrays and ensure 1D
        y_true = np.atleast_1d(np.array(y_true))
        y_pred = np.atleast_1d(np.array(y_pred))
        X = np.atleast_2d(np.array(X))

        # Validate shapes
        if y_true.shape[0] != y_pred.shape[0]:
            print(f"Warning: Shape mismatch - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
            return

        if y_true.shape[0] != X.shape[0]:
            print(f"Warning: Shape mismatch - labels: {y_true.shape[0]}, samples: {X.shape[0]}")
            return

        # Find misclassified indices
        try:
            misclassified_idx = np.where(y_true != y_pred)[0]
        except Exception as e:
            print(f"Error finding misclassifications: {e}")
            print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            return

        if len(misclassified_idx) == 0:
            print("\n" + "=" * 70)
            print("✓ No misclassifications found! Perfect classification!")
            print("=" * 70)
            return

        print(f"\nFound {len(misclassified_idx)} misclassified samples:")
        print("=" * 70)

        # Ensure X is a proper array
        X = np.atleast_2d(X)

        misclass_df = pd.DataFrame({
            'Sample Index': misclassified_idx,
            'True Label': [target_names[y_true[i]] for i in misclassified_idx],
            'Predicted Label': [target_names[y_pred[i]] for i in misclassified_idx]
        })

        print(misclass_df.to_string(index=False))

        # Visualize misclassified samples
        if len(misclassified_idx) > 0:
            n_plots = min(len(misclassified_idx), 4)
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

            # Handle single plot case
            if n_plots == 1:
                axes = [axes]

            for plot_idx, sample_idx in enumerate(misclassified_idx[:4]):
                ax = axes[plot_idx]

                # Get sample features - handle both 1D and 2D cases
                if X.ndim == 1:
                    sample_features = X
                elif X.ndim == 2:
                    sample_features = X[sample_idx]
                else:  # 3D (reshaped for CNN)
                    sample_features = X[sample_idx].flatten()

                # Ensure feature_names matches sample_features length
                n_features = len(sample_features)
                display_features = feature_names[:n_features] if len(feature_names) > n_features else feature_names

                ax.bar(range(len(sample_features)), sample_features,
                      color='coral', alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(sample_features)))
                ax.set_xticklabels(display_features, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Feature Value', fontsize=10)
                ax.set_title(f'Sample {sample_idx}\nTrue: {target_names[y_true[sample_idx]]}\n'
                           f'Pred: {target_names[y_pred[sample_idx]]}',
                           fontsize=10, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig('../results/misclassified_samples.png', dpi=300, bbox_inches='tight')
            plt.show()


def generate_comprehensive_report(results, feature_importance, loocv_accuracy):
    """
    Generate a comprehensive text report
    """
    report = []
    report.append("=" * 80)
    report.append("IRIS CLASSIFICATION WITH CNN - COMPREHENSIVE REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("1. EXECUTIVE SUMMARY")
    report.append("-" * 80)
    best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
    report.append(f"Best Performing Model: {best_config[0]}")
    report.append(f"Test Set Accuracy: {best_config[1]['accuracy']*100:.2f}%")
    report.append(f"LOOCV Accuracy: {loocv_accuracy*100:.2f}%")
    report.append("")

    report.append("2. MODEL CONFIGURATION RESULTS")
    report.append("-" * 80)
    for config_name, result in sorted(results.items(),
                                     key=lambda x: x[1]['accuracy'],
                                     reverse=True):
        report.append(f"{config_name:40s} Accuracy: {result['accuracy']*100:6.2f}%")
    report.append("")

    report.append("3. FEATURE IMPORTANCE ANALYSIS")
    report.append("-" * 80)
    report.append("Features ranked by importance (as learned by CNN):")
    for idx, row in feature_importance.iterrows():
        report.append(f"  {idx+1}. {row['Feature']:30s} Score: {row['Importance Score']:.4f}")
    report.append("")

    report.append("4. KEY FINDINGS")
    report.append("-" * 80)
    report.append("• The CNN successfully learned to classify IRIS species with high accuracy")
    report.append("• Different hyperparameter configurations showed varying performance")
    report.append("• LOOCV provided a robust estimate of generalization performance")
    report.append("• Feature importance analysis revealed which botanical measurements")
    report.append("  are most discriminative for species classification")
    report.append("")

    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open('../results/analysis_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text


if __name__ == "__main__":
    print("Advanced Analysis Module")
    print("This module provides additional visualization and analysis tools")
    print("Import and use the classes in your main script")