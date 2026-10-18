import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_metrics():
    """Load metrics from the metrics.json file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(script_dir, "runs", "eval", "exp", "metrics.json")
    print(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_evaluation_summary(metrics):
    """Create a bar chart visualization of the evaluation metrics."""

    # Extract metrics
    per_digit_accuracy = metrics['per_digit_accuracy'] * 100  # Convert to percentage
    exact_match_accuracy = metrics['full_5digit_exact_match_accuracy'] * 100  # Convert to percentage
    precision_macro = metrics['precision_macro']
    recall_macro = metrics['recall_macro']
    f1_macro = metrics['f1_macro']

    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Segmented-Digit CNN -- Evaluation Summary', fontsize=16, fontweight='bold')

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
    output_path = os.path.join(script_dir, "runs", "eval", "exp", "evaluation_summary_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    try:
        metrics = load_metrics()
        plot_evaluation_summary(metrics)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()