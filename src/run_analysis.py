"""
Main Execution Script - IRIS CNN Classification Project
This script runs the complete analysis pipeline
"""

import sys
import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create results directory
os.makedirs('../results', exist_ok=True)

# Import main modules
from iris_cnn_analysis import *
from advanced_analysis_CNN import *

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_complete_analysis():
    """
    Run the complete analysis pipeline for IRIS classification
    """
    start_time = time.time()

    # Print system information
    print_header("SYSTEM INFORMATION")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: Available ({len(gpus)} device(s))")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("GPU: Not available (using CPU)")

    # STEP 1: Load and explore data
    print_header("STEP 1: DATA LOADING AND EXPLORATION")

    data_processor = IrisDataProcessor()
    X, y, feature_names, target_names = data_processor.load_and_prepare_data()
    X_processed, y_encoded, y_original = data_processor.preprocess_data(X, y)

    input_shape = (X_processed.shape[1], 1)
    print(f"\nProcessed Input Shape: {input_shape}")

    # Visualize dataset
    print("\nGenerating dataset visualizations...")
    visualizer = AdvancedVisualizer()

    df = visualizer.plot_dataset_distribution(X, y_original, feature_names, target_names)
    visualizer.plot_correlation_matrix(X, feature_names)
    visualizer.plot_pairwise_features(X, y_original, feature_names, target_names)
    visualizer.plot_dimensionality_reduction(X, y_original, target_names)

    # STEP 2: Compare different CNN configurations
    print_header("STEP 2: COMPARING CNN CONFIGURATIONS")

    results = compare_model_configurations(
        X_processed, y_encoded, y_original, input_shape, target_names
    )

    # STEP 3: Detailed performance analysis
    print_header("STEP 3: DETAILED PERFORMANCE ANALYSIS")

    perf_analyzer = PerformanceAnalyzer()
    perf_analyzer.compare_configurations_detailed(results)

    # STEP 4: Select best model and perform LOOCV
    print_header("STEP 4: LEAVE-ONE-OUT CROSS-VALIDATION")

    best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_config_name = best_config[0]
    best_model = best_config[1]['model']

    print(f"Best Configuration: {best_config_name}")
    print(f"Test Set Accuracy: {best_config[1]['accuracy']*100:.2f}%")

    # Perform LOOCV
    print("\nPerforming LOOCV (this may take a few minutes)...")
    model_builder = CNNModelBuilder(input_shape)

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

    # Evaluate LOOCV
    loocv_accuracy = ModelEvaluator.calculate_metrics(
        loocv_true, loocv_predictions, target_names
    )

    ModelEvaluator.plot_confusion_matrix(
        loocv_true, loocv_predictions, target_names,
        f"{best_config_name} - LOOCV",
        save_path="../results/confusion_matrix_loocv.png"
    )

    # Analyze misclassifications
    print("\nAnalyzing misclassifications...")
    # Ensure arrays are numpy arrays
    loocv_true_arr = np.array(loocv_true)
    loocv_predictions_arr = np.array(loocv_predictions)
    perf_analyzer.analyze_misclassifications(
        loocv_true_arr, loocv_predictions_arr, X, feature_names, target_names
    )

    # STEP 5: Feature importance and model internals
    print_header("STEP 5: FEATURE IMPORTANCE AND MODEL ANALYSIS")

    feature_importance = FeatureAnalyzer.analyze_feature_importance(
        best_model, X_processed, feature_names
    )

    # Analyze model internals
    print("\nAnalyzing model internals...")
    model_analyzer = ModelAnalyzer()

    # Visualize layer outputs for a sample
    sample_idx = 0
    print(f"\nVisualizing layer activations for sample {sample_idx} "
          f"({target_names[y_original[sample_idx]]})...")
    model_analyzer.visualize_layer_outputs(best_model, X_processed[sample_idx])

    # Visualize filter weights
    print("\nVisualizing convolutional filter weights...")
    model_analyzer.analyze_filter_weights(best_model)

    # Plot model architecture
    print("\nGenerating model architecture diagram...")
    model_analyzer.plot_model_architecture(best_model)

    # STEP 6: Generate comprehensive report
    print_header("STEP 6: GENERATING COMPREHENSIVE REPORT")

    report_text = generate_comprehensive_report(results, feature_importance, loocv_accuracy)

    # STEP 7: Summary statistics
    print_header("STEP 7: FINAL SUMMARY")

    elapsed_time = time.time() - start_time

    print(f"Analysis completed successfully!")
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    print(f"\nBest Model: {best_config_name}")
    print(f"Test Accuracy: {best_config[1]['accuracy']*100:.2f}%")
    print(f"LOOCV Accuracy: {loocv_accuracy*100:.2f}%")
    print(f"\nTop 3 Most Important Features:")
    for idx in range(min(3, len(feature_importance))):
        row = feature_importance.iloc[idx]
        print(f"  {idx+1}. {row['Feature']}: {row['Importance Score']:.4f}")

    print("\n" + "=" * 80)
    print("ALL RESULTS SAVED TO: ../results/")
    print("-" * 80)
    saved_files = [
        "iris_cnn_complete.py (main script)",
        "advanced_analysis.py (analysis module)",
        "dataset_distribution.png",
        "correlation_matrix.png",
        "pairwise_features.png",
        "dimensionality_reduction.png",
        "feature_importance.png",
        "comprehensive_comparison.png",
        "confusion_matrix_loocv.png",
        "layer_activations.png",
        "filter_weights.png",
        "model_architecture.png",
        "analysis_report.txt",
        "training_history_*.png (multiple files)",
        "confusion_matrix_*.png (multiple files)"
    ]

    for filename in saved_files:
        print(f"  ✓ {filename}")

    print("=" * 80)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Configuration': list(results.keys()),
        'Test Accuracy (%)': [r['accuracy']*100 for r in results.values()]
    }).sort_values('Test Accuracy (%)', ascending=False)

    summary_df['LOOCV Accuracy (%)'] = loocv_accuracy * 100

    print("\n" + "=" * 80)
    print("CONFIGURATION PERFORMANCE SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return {
        'results': results,
        'best_model': best_model,
        'best_config_name': best_config_name,
        'loocv_accuracy': loocv_accuracy,
        'feature_importance': feature_importance,
        'execution_time': elapsed_time
    }


if __name__ == "__main__":
    import sklearn

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         DEEP LEARNING LAB PROJECT - PART 1                               ║
    ║         IRIS Classification with Convolutional Neural Networks           ║
    ║                                                                          ║
    ║         University of Business and Technology (UBT)                      ║
    ║         Winter Term 2025/2026                                            ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        analysis_results = run_complete_analysis()

        print("\n" + "🎉" * 40)
        print("\n✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("\n" + "🎉" * 40)

    except Exception as e:
        print(f"\n❌ Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()