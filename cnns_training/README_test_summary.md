# Test Set Evolution Summary

This directory contains scripts for generating evolution summaries and visualizations of your CNN model's performance on the test set.

## Files

1. `test_evolution_summary.py` - Text-based summary of test metrics
2. `plot_test_evolution_summary.py` - Visual plot of test metrics (matches requested format)
3. `test_evaluation_summary_plot.png` - Generated visualization (after running the plot script)

## Test Set Details
- **Images**: 212
- **Digit crops**: 212 × 5 = 1,060 samples
- **Source**: `confusion_matrix_metrics.json` (test section)

## Key Metrics (from test set)
- Per-digit accuracy: 74.81%
- 5-digit exact match accuracy: 51.89%
- Exact matches: 110 / 212 images
- Macro-averaged precision: 0.7479
- Macro-averaged recall: 0.6842
- Macro-averaged F1-score: 0.7084

## Usage
```bash
# Get text summary
python test_evolution_summary.py

# Generate visualization plot
python plot_test_evolution_summary.py
```