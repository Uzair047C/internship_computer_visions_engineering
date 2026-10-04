#!/usr/bin/env python3
"""
CNN Confusion Matrix Evaluation Script

This script tests the trained CNN model on both validation and test sets,
generates confusion matrices, and provides detailed metrics for both splits.

Dataset Structure (from data_set_generator_for_cnns_only):
- Train: 1,270 images (6,350 digit crops)
- Validation: 173 images (865 digit crops)
- Test: 212 images (1,060 digit crops)

Outputs:
- Confusion matrix plots (counts and normalized)
- Classification reports
- Per-digit accuracy analysis
- Full 5-digit exact-match accuracy
- Comparison with paper claims
"""

import os
import json
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================
NUM_DIGITS = 5
NUM_CLASSES = 10
DIGIT_HEIGHT = 32
DIGIT_WIDTH = 32
BATCH_SIZE = 32

# Paths - Dataset generator structure
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_set_generator_for_cnns_only", "data_set")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
VALID_IMG_DIR = os.path.join(BASE_DIR, "valid")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VALID_CSV = os.path.join(BASE_DIR, "valid.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "single_digit_model_final.keras")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "runs", "eval", "exp")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SEGMENTATION FUNCTIONS (matching vgg_net.ipynb)
# ============================================================
MIN_COMPONENT_AREA_FRAC = 0.01
MAX_COMPONENT_AREA_FRAC = 0.60
BOX_PADDING_PX = 2


def _binarize_for_segmentation(gray_img):
    """Denoise + threshold for digit segmentation."""
    denoised = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def segment_digit_boxes(gray_img, num_digits=NUM_DIGITS):
    """Segment image into digit bounding boxes."""
    h_img, w_img = gray_img.shape[:2]
    img_area = h_img * w_img
    binary = _binarize_for_segmentation(gray_img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        area_frac = area / img_area
        if MIN_COMPONENT_AREA_FRAC <= area_frac <= MAX_COMPONENT_AREA_FRAC:
            boxes.append((x, y, w, h))
    if len(boxes) == num_digits:
        boxes.sort(key=lambda b: b[0])
        return boxes
    if len(boxes) > num_digits:
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        boxes = boxes[:num_digits]
        boxes.sort(key=lambda b: b[0])
        return boxes
    # Fallback: equal-width columns
    col_width = w_img / num_digits
    return [(int(i * col_width), 0, int(col_width), h_img) for i in range(num_digits)]


def extract_digit_crops(gray_img, num_digits=NUM_DIGITS, target_size=(DIGIT_HEIGHT, DIGIT_WIDTH)):
    """Extract and resize digit crops."""
    h_img, w_img = gray_img.shape[:2]
    boxes = segment_digit_boxes(gray_img, num_digits=num_digits)
    crops = []
    for (x, y, w, h) in boxes:
        x0 = max(0, x - BOX_PADDING_PX)
        y0 = max(0, y - BOX_PADDING_PX)
        x1 = min(w_img, x + w + BOX_PADDING_PX)
        y1 = min(h_img, y + h + BOX_PADDING_PX)
        crop = gray_img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        crop = cv2.resize(crop, (target_size[1], target_size[0]))
        crop = crop.astype(np.float32) / 255.0
        crops.append(np.expand_dims(crop, axis=-1))
    return crops


def build_digit_dataset(df, img_dir, num_digits=NUM_DIGITS):
    """Build per-digit dataset from CSV."""
    X, y, group_ids = [], [], []
    for group_id, (_, row) in enumerate(df.iterrows()):
        img_path = os.path.join(img_dir, str(row['image']))
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            continue
        padded_label = str(int(row['label'])).zfill(num_digits)
        crops = extract_digit_crops(gray_img, num_digits=num_digits)
        for digit_char, crop in zip(padded_label, crops):
            X.append(crop)
            y.append(int(digit_char))
            group_ids.append(group_id)
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    group_ids = np.array(group_ids, dtype=np.int64)
    return X, y, group_ids


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def evaluate_split(model, X, y, group_ids, split_name, save_prefix):
    """Evaluate model on a single split (valid/test)."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {split_name.upper()} SET")
    print(f"{'='*70}")
    print(f"Digit crops: {X.shape[0]}")
    print(f"Images: {group_ids.max() + 1}")

    # Run inference
    preds = model.predict(X, verbose=1, batch_size=BATCH_SIZE)
    pred_labels = np.argmax(preds, axis=1)

    # Per-digit accuracy
    per_digit_acc = np.mean(pred_labels == y)
    print(f"\nPer-digit accuracy: {per_digit_acc:.4f} ({per_digit_acc*100:.2f}%)")

    # Full 5-digit exact-match accuracy
    num_groups = group_ids.max() + 1
    exact_matches = 0
    for g in range(num_groups):
        idx = np.where(group_ids == g)[0]
        if len(idx) == NUM_DIGITS:
            if np.array_equal(pred_labels[idx], y[idx]):
                exact_matches += 1
    exact_match_acc = exact_matches / num_groups
    print(f"Full 5-digit exact-match accuracy: {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)")
    print(f"Exact matches: {exact_matches} / {num_groups}")

    # Classification report
    print(f"\n--- Classification Report ({split_name}) ---")
    report = classification_report(y, pred_labels, digits=4, target_names=[str(i) for i in range(10)])
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y, pred_labels, labels=list(range(10)))
    print(f"\n--- Confusion Matrix ({split_name}, Counts) ---")
    print(cm)
    print(f"Total samples: {cm.sum()}")

    # Normalized confusion matrix (recall per class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    print(f"\n--- Normalized Confusion Matrix ({split_name}, Recall) ---")
    for i in range(10):
        print(f"Class {i}: {cm_norm[i]}")

    # Per-class support
    supports = cm.sum(axis=1)
    print(f"\nPer-class support: {supports}")
    print(f"Total support: {supports.sum()}")

    # Save confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(10), yticklabels=range(10))
    axes[0].set_title(f'Confusion Matrix (Counts) - {split_name.capitalize()} Set')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=range(10), yticklabels=range(10), vmin=0, vmax=1)
    axes[1].set_title(f'Confusion Matrix (Normalized/Recall) - {split_name.capitalize()} Set')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{save_prefix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()

    # Return metrics for comparison
    return {
        'split': split_name,
        'num_images': int(num_groups),
        'num_digit_crops': int(X.shape[0]),
        'per_digit_accuracy': float(per_digit_acc),
        'exact_match_accuracy': float(exact_match_acc),
        'exact_matches': int(exact_matches),
        'confusion_matrix_counts': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'per_class_support': supports.tolist(),
        'classification_report': report
    }


def save_metrics_json(metrics_valid, metrics_test):
    """Save combined metrics to JSON."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_path': MODEL_PATH,
        'dataset_split': {
            'train_images': 1270,
            'valid_images': 173,
            'test_images': 212
        },
        'validation': metrics_valid,
        'test': metrics_test
    }

    json_path = os.path.join(OUTPUT_DIR, "confusion_matrix_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")

    return json_path


def print_summary(metrics_valid, metrics_test):
    """Print final summary comparing with paper claims."""
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\nDataset Split (from notebook):")
    print(f"  Train: 1,270 images (6,350 digit crops)")
    print(f"  Valid: 173 images   (865 digit crops)")
    print(f"  Test:  212 images   (1,060 digit crops)")

    print(f"\nPaper Table I (rounded):")
    print(f"  1,200 / 220 / 212")

    print(f"\nValidation Set Results:")
    print(f"  Per-digit accuracy: {metrics_valid['per_digit_accuracy']:.4f} ({metrics_valid['per_digit_accuracy']*100:.2f}%)")
    print(f"  Full 5-digit exact-match: {metrics_valid['exact_match_accuracy']:.4f} ({metrics_valid['exact_match_accuracy']*100:.2f}%)")
    print(f"  Exact matches: {metrics_valid['exact_matches']} / {metrics_valid['num_images']}")

    print(f"\nTest Set Results:")
    print(f"  Per-digit accuracy: {metrics_test['per_digit_accuracy']:.4f} ({metrics_test['per_digit_accuracy']*100:.2f}%)")
    print(f"  Full 5-digit exact-match: {metrics_test['exact_match_accuracy']:.4f} ({metrics_test['exact_match_accuracy']*100:.2f}%)")
    print(f"  Exact matches: {metrics_test['exact_matches']} / {metrics_test['num_images']}")

    print(f"\nPaper Claims (FIXED):")
    print(f"  CNN per-digit accuracy: 76.32%")
    print(f"  CNN full 5-digit exact-match: 56.60% (120/212)")
    print(f"  PaddleOCR full 5-digit exact-match: 94.71% (197/208 same-length)")

    print(f"\nVerification:")
    print(f"  Test set size: 212 images (NOT 180 as originally claimed)")
    print(f"  Test digit crops: 1,060 (212 × 5)")
    print(f"  Confusion matrix computed on: {metrics_test['num_digit_crops']} digit instances")
    print(f"  Full-sequence accuracy computed on: {metrics_test['num_images']} images")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("CNN CONFUSION MATRIX EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Load CSVs
    print("\nLoading CSV files...")
    df_valid = pd.read_csv(VALID_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Convert labels to int
    df_valid['label'] = df_valid['label'].astype(int)
    df_test['label'] = df_test['label'].astype(int)

    print(f"Valid CSV rows: {len(df_valid)}")
    print(f"Test CSV rows: {len(df_test)}")

    # Build datasets
    print("\nBuilding validation dataset (segmenting into digit crops)...")
    X_valid, y_valid, valid_group_ids = build_digit_dataset(df_valid, VALID_IMG_DIR)
    print(f"Valid digit crops: {X_valid.shape[0]} (from {valid_group_ids.max() + 1} images)")

    print("\nBuilding test dataset (segmenting into digit crops)...")
    X_test, y_test, test_group_ids = build_digit_dataset(df_test, TEST_IMG_DIR)
    print(f"Test digit crops: {X_test.shape[0]} (from {test_group_ids.max() + 1} images)")

    # Evaluate both splits
    metrics_valid = evaluate_split(model, X_valid, y_valid, valid_group_ids, "validation", "valid")
    metrics_test = evaluate_split(model, X_test, y_test, test_group_ids, "test", "test")

    # Save combined metrics
    save_metrics_json(metrics_valid, metrics_test)

    # Print summary
    print_summary(metrics_valid, metrics_test)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()