import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_test_metrics():
    """Load test metrics from the confusion_matrix_metrics.json file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(script_dir, "runs", "eval", "exp", "confusion_matrix_metrics.json")
    print(f"Loading test metrics from: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if 'test' not in metrics:
        raise ValueError("No 'test' data found in metrics file")

    return metrics['test']

def plot_test_evaluation_summary(test_data):
    """Create a bar chart visualization of the test evaluation metrics."""

    # Extract test metrics
    per_digit_accuracy = test_data['per_digit_accuracy'] * 100  # Convert to percentage
    exact_match_accuracy = test_data['exact_match_accuracy'] * 100  # Convert to percentage

    # For macro-averaged metrics, we need to extract from classification report or calculate
    # Since we have the classification report string, let's parse it or use the normalized confusion matrix
    # Actually, looking at the data, we can calculate macro averages from the classification report
    # But easier: the confusion_matrix_normalized gives us what we need

    # Let's extract from the classification report for simplicity
    # The classification report has: precision, recall, f1-score for each class and averages
    report_lines = test_data['classification_report'].split('\n')

    # Extract macro averages from classification report
    # Format has consistent spacing, so we can parse it reliably
    precision_macro = recall_macro = f1_macro = 0
    for line in report_lines:
        if 'macro avg' in line:
            # Split by whitespace and filter out empty strings
            parts = [p for p in line.split() if p.strip()]
            # Find the index of 'macro' and 'avg' to get the numbers after them
            try:
                macro_index = parts.index('macro')
                if macro_index + 2 < len(parts) and parts[macro_index + 1] == 'avg':
                    precision_macro = float(parts[macro_index + 2])
                    recall_macro = float(parts[macro_index + 3])
                    f1_macro = float(parts[macro_index + 4])
                    break
            except (ValueError, IndexError):
                # Fallback: try to find numbers after 'macro avg' as a phrase
                import re
                # Look for pattern: macro avg followed by numbers
                match = re.search(r'macro\s+avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    precision_macro = float(match.group(1))
                    recall_macro = float(match.group(2))
                    f1_macro = float(match.group(3))
                    break

    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Test Set -- Evaluation Summary', fontsize=16, fontweight='bold')

    # Plot 1: Overall Accuracy
    accuracy_labels = ['Per-digit accuracy', '5-digit exact match']
    accuracy_values = [per_digit_accuracy, exact_match_accuracy]
    accuracy_colors = ['#3498db', '#2ecc71']

    bars1 = ax1.bar(accuracy_labels, accuracy_values, color=accuracy_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, value in zip(bars1, accuracy_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Macro-averaged Metrics
    macro_labels = ['Precision', 'Recall', 'F1-Score']
    macro_values = [precision_macro, recall_macro, f1_macro]
    macro_colors = ['#e74c3c', '#f39c12', '#9b59b6']

    bars2 = ax2.bar(macro_labels, macro_values, color=macro_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Macro-averaged Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars2, macro_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "runs", "eval", "exp", "test_evaluation_summary_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Test evaluation plot saved to: {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    try:
        test_data = load_test_metrics()
        plot_test_evaluation_summary(test_data)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()