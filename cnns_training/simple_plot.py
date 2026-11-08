# Simple confusion matrix plotting using saved metrics
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 70)
print("PLOTTING CONFUSION MATRIX FROM SAVED METRICS")
print("=" * 70)

# Load the confusion matrix metrics
metrics_path = os.path.join(os.getcwd(), "runs", "eval", "exp", "confusion_matrix_metrics.json")
print(f"Loading metrics from: {metrics_path}")

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract test confusion matrix
test_data = metrics['test']
conf_matrix = np.array(test_data['confusion_matrix_counts'])
conf_matrix_norm = np.array(test_data['confusion_matrix_normalized'])
class_names = [str(i) for i in range(10)]

print(f"Test set: {test_data['num_images']} images × 5 digits = {test_data['num_digit_crops']} samples")

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot counts
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('Confusion Matrix (Counts)\nTest Set: 212 images × 5 digits = 1,060 samples')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Plot normalized
sns.heatmap(conf_matrix_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalized / Recall)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.getcwd(), "runs", "eval", "exp", "confusion_matrix_from_notebook.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

# Show the plot
plt.show()

# Print metrics
print("\n" + "=" * 70)
print("METRICS FROM SAVED DATA")
print("=" * 70)
print(f"Per-digit accuracy: {test_data['per_digit_accuracy']:.4f} ({test_data['per_digit_accuracy']*100:.2f}%)")
print(f"Exact match accuracy: {test_data['exact_match_accuracy']:.4f} ({test_data['exact_match_accuracy']*100:.2f}%)")
print(f"Exact matches: {test_data['exact_matches']} / {test_data['num_digit_crops']//5} images")

# Show classification report
print("\nClassification Report:")
print(test_data['classification_report'])