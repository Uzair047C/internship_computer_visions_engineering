import os
import json

def load_metrics():
    # Prefer the confusion matrix metrics file as it has validation and test splits
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(script_dir, "runs", "eval", "exp", "confusion_matrix_metrics.json")
    if not os.path.exists(metrics_path):
        # Fallback to the regular metrics file
        metrics_path = os.path.join(script_dir, "runs", "eval", "exp", "metrics.json")
    print(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def print_summary(metrics):
    print("=" * 80)
    print("MODEL EVOLUTION SUMMARY")
    print("=" * 80)

    # Check if we have validation and test data (from confusion_matrix_metrics.json)
    if 'validation' in metrics and 'test' in metrics:
        validation_data = metrics['validation']
        test_data = metrics['test']

        print(f"Validation Set: {validation_data['num_images']} images × 5 digits = {validation_data['num_digit_crops']} samples")
        print(f"Test Set: {test_data['num_images']} images × 5 digits = {test_data['num_digit_crops']} samples")

        print("\n" + "-" * 80)
        print("VALIDATION METRICS")
        print("-" * 80)
        print(f"Per-digit accuracy: {validation_data['per_digit_accuracy']:.4f} ({validation_data['per_digit_accuracy']*100:.2f}%)")
        print(f"Exact match accuracy: {validation_data['exact_match_accuracy']:.4f} ({validation_data['exact_match_accuracy']*100:.2f}%)")
        print(f"Exact matches: {validation_data['exact_matches']} / {validation_data['num_digit_crops']//5} images")

        print("\n" + "-" * 80)
        print("TEST METRICS")
        print("-" * 80)
        print(f"Per-digit accuracy: {test_data['per_digit_accuracy']:.4f} ({test_data['per_digit_accuracy']*100:.2f}%)")
        print(f"Exact match accuracy: {test_data['exact_match_accuracy']:.4f} ({test_data['exact_match_accuracy']*100:.2f}%)")
        print(f"Exact matches: {test_data['exact_matches']} / {test_data['num_digit_crops']//5} images")

        # Calculate improvement if we had previous data? We don't, so just note current state.
        print("\n" + "=" * 80)
        print("NOTE: This summary reflects the current model state.")
        print("For evolution tracking, consider saving metrics after each training epoch.")
        print("=" * 80)
    else:
        # Fallback to the overall metrics (from metrics.json)
        print("Overall metrics (no validation/test split):")
        print(f"Timestamp: {metrics.get('timestamp', 'N/A')}")
        print(f"Per-digit accuracy: {metrics.get('per_digit_accuracy', 0):.4f} ({metrics.get('per_digit_accuracy', 0)*100:.2f}%)")
        print(f"Full 5-digit exact match accuracy: {metrics.get('full_5digit_exact_match_accuracy', 0):.4f} ({metrics.get('full_5digit_exact_match_accuracy', 0)*100:.2f}%)")
        print(f"Macro precision: {metrics.get('precision_macro', 0):.4f}")
        print(f"Macro recall: {metrics.get('recall_macro', 0):.4f}")
        print(f"Macro F1: {metrics.get('f1_macro', 0):.4f}")

        if 'per_class' in metrics:
            print("\nPer-class metrics:")
            for class_id, class_metrics in metrics['per_class'].items():
                print(f"  Class {class_id}: precision={class_metrics['precision']:.4f}, recall={class_metrics['recall']:.4f}, f1={class_metrics['f1']:.4f}, support={class_metrics['support']}")

        print("\n" + "=" * 80)
        print("NOTE: For validation/test split metrics, ensure confusion_matrix_metrics.json is present.")
        print("=" * 80)

if __name__ == "__main__":
    try:
        metrics = load_metrics()
        print_summary(metrics)
    except Exception as e:
        print(f"Error loading metrics: {e}")