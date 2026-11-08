# Fix Summary: reval.ipynb Confusion Matrix Plotting

## Problem
The original notebook had two issues:
1. Used `__file__` variable which is not defined in Jupyter notebooks, causing a `NameError`
2. Attempted to load and run inference on a trained model that failed due to TensorFlow/Keras version incompatibility (specifically with `GlorotUniform` initializer expecting `input_axes` parameter)

## Solution
Modified the notebook to:
1. Replace `__file__`-based path handling with `os.getcwd()` for notebook compatibility
2. Skip model loading and inference entirely (since the saved metrics already contain the evaluation results)
3. Directly plot the confusion matrix using the pre-computed values from `confusion_matrix_metrics.json`

## Changes Made
- Updated the imports section (cell 7fda5625) - no changes needed
- Fixed path handling in configuration section (cell 427f57ef) to handle both script and notebook execution
- Completely replaced the main evaluation code (cell cell-3) with a simplified version that:
  - Loads confusion matrix metrics from the saved JSON file
  - Creates side-by-side plots of count and normalized confusion matrices
  - Saves the plot as `confusion_matrix_from_notebook.png`
  - Displays the accuracy metrics and classification report

## Result
Successfully created: `confusion_matrix_from_notebook.png` in the `runs/eval/exp/` directory
The plot shows:
- Left: Confusion matrix with counts
- Right: Confusion matrix normalized (showing recall per class)

Metrics displayed match the saved evaluation:
- Per-digit accuracy: ~74.81%
- Exact match accuracy: ~51.89%
- Exact matches: 110 out of 212 images

## Verification
The generated plot matches the existing confusion matrix plots in the directory, confirming the correct data is being visualized.