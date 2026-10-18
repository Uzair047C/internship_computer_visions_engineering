import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the confusion matrix metrics
script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(script_dir, "runs", "eval", "exp", "confusion_matrix_metrics.json")
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract test confusion matrix (the most recent evaluation)
test_data = metrics['test']
conf_matrix = np.array(test_data['confusion_matrix_counts'])
conf_matrix_norm = np.array(test_data['confusion_matrix_normalized'])
class_names = [str(i) for i in range(10)]

print("Test Confusion Matrix (Counts):")
print(conf_matrix)
print(f"\nTotal samples: {np.sum(conf_matrix)}")

print("\nTest Confusion Matrix (Normalized):")
print(conf_matrix_norm)

# Create the plot with larger figure size for better visibility
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot counts
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 20})  # Increased annotation size
axes[0].set_title('Confusion Matrix (Counts)\nTest Set: 212 images × 5 digits = 1,060 samples', fontsize=24)
axes[0].set_xlabel('Predicted', fontsize=20)
axes[0].set_ylabel('True', fontsize=20)
axes[0].tick_params(axis='both', which='major', labelsize=16)

# Plot normalized
sns.heatmap(conf_matrix_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1,
            annot_kws={"size": 20})  # Increased annotation size
axes[1].set_title('Confusion Matrix (Normalized / Recall)', fontsize=24)
axes[1].set_xlabel('Predicted', fontsize=20)
axes[1].set_ylabel('True', fontsize=20)
axes[1].tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()

# Save the plot with higher DPI for better quality
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "runs", "eval", "exp", "confusion_matrix_replotted.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Show the plot
plt.show()

# Print some metrics from the JSON
print("\n" + "="*50)
print("METRICS FROM JSON:")
print("="*50)
print(f"Per-digit accuracy: {test_data['per_digit_accuracy']:.4f} ({test_data['per_digit_accuracy']*100:.2f}%)")
print(f"Exact match accuracy: {test_data['exact_match_accuracy']:.4f} ({test_data['exact_match_accuracy']*100:.2f}%)")
print(f"Exact matches: {test_data['exact_matches']} / {test_data['num_digit_crops']//5} images")