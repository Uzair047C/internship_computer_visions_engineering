import os
import json

def load_test_metrics():
    """Load and extract test metrics from confusion_matrix_metrics.json."""
    metrics_path = os.path.join(os.getcwd(), "runs", "eval", "exp", "confusion_matrix_metrics.json")
    print(f"Loading test metrics from: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if 'test' not in metrics:
        raise ValueError("No 'test' data found in metrics file")

    return metrics['test']

def print_test_summary(test_data):
    """Print evolution summary for test data only."""
    print("=" * 60)
    print("TEST DATA EVOLUTION SUMMARY")
    print("=" * 60)

    print(f"Dataset: {test_data['num_images']} images × 5 digits = {test_data['num_digit_crops']} samples")
    print()

    print("TEST METRICS:")
    print("-" * 40)
    print(f"Per-digit accuracy:     {test_data['per_digit_accuracy']:.4f} ({test_data['per_digit_accuracy']*100:.2f}%)")
    print(f"Exact match accuracy:   {test_data['exact_match_accuracy']:.4f} ({test_data['exact_match_accuracy']*100:.2f}%)")
    print(f"Exact matches:          {test_data['exact_matches']} / {test_data['num_digit_crops']//5} images")
    print()

    # Show per-class performance if available
    if 'classification_report' in test_data:
        print("Per-class performance (from classification report):")
        print("-" * 40)
        # Extract key lines from classification report
        report_lines = test_data['classification_report'].split('\n')
        for line in report_lines[2:12]:  # Skip header, take per-class lines
            if line.strip() and not line.startswith('accuracy') and not line.startswith('macro') and not line.startswith('weighted'):
                print(line)

    print()
    print("=" * 60)
    print("Note: This summary focuses exclusively on the test set (212 images)")
    print("For comparison with validation, see the full evolution summary.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_data = load_test_metrics()
        print_test_summary(test_data)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()