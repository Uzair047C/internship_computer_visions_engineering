#!/usr/bin/env python3
"""
Re-evaluate the CNN model on the test set and verify confusion matrix numbers.
This script loads the trained model, runs inference on the test set,
and compares results with the saved metrics.json.
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

# ============================================================
# CONFIGURATION (matching the notebook)
# ============================================================
NUM_DIGITS = 5
NUM_CLASSES = 10
DIGIT_HEIGHT = 32
DIGIT_WIDTH = 32

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_set_generator_for_cnns_only", "data_set")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
VALID_IMG_DIR = os.path.join(BASE_DIR, "valid")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VALID_CSV = os.path.join(BASE_DIR, "valid.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "single_digit_model_final.keras")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "runs", "eval", "exp", "metrics.json")

# ============================================================
# SEGMENTATION FUNCTIONS (from notebook)
# ============================================================
MIN_COMPONENT_AREA_FRAC = 0.01
MAX_COMPONENT_AREA_FRAC = 0.60
BOX_PADDING_PX = 2

def _binarize_for_segmentation(gray_img):
    denoised = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def segment_digit_boxes(gray_img, num_digits=NUM_DIGITS):
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
    col_width = w_img / num_digits
    return [(int(i * col_width), 0, int(col_width), h_img) for i in range(num_digits)]

def extract_digit_crops(gray_img, num_digits=NUM_DIGITS, target_size=(DIGIT_HEIGHT, DIGIT_WIDTH)):
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
# MAIN EVALUATION
# ============================================================
def main():
    print("=" * 70)
    print("CNN RE-EVALUATION SCRIPT")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    model.summary()

    # Load test CSV
    print(f"\nLoading test data from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)
    print(f"Test CSV rows: {len(df_test)}")

    # Check images exist
    missing = 0
    for _, row in df_test.iterrows():
        img_path = os.path.join(TEST_IMG_DIR, str(row['image']))
        if not os.path.exists(img_path):
            missing += 1
    print(f"Missing images: {missing}")
    print(f"Available test images: {len(df_test) - missing}")

    # Build test dataset (digit crops)
    print("\nBuilding test dataset (segmenting into digit crops)...")
    X_test, y_test, test_group_ids = build_digit_dataset(df_test, TEST_IMG_DIR)
    print(f"Test digit crops: {X_test.shape[0]} (from {test_group_ids.max() + 1} images)")
    print(f"Expected: 212 images × 5 digits = 1,060 crops")

    # Load saved metrics for comparison
    with open(METRICS_PATH, 'r') as f:
        saved_metrics = json.load(f)

    print("\n" + "=" * 70)
    print("SAVED METRICS (from metrics.json)")
    print("=" * 70)
    print(f"Per-digit accuracy: {saved_metrics['per_digit_accuracy']:.4f} ({saved_metrics['per_digit_accuracy']*100:.2f}%)")
    print(f"Full 5-digit exact-match accuracy: {saved_metrics['full_5digit_exact_match_accuracy']:.4f} ({saved_metrics['full_5digit_exact_match_accuracy']*100:.2f}%)")
    print(f"Macro Precision: {saved_metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {saved_metrics['recall_macro']:.4f}")
    print(f"Macro F1: {saved_metrics['f1_macro']:.4f}")
    print(f"Support per class: {[saved_metrics['per_class'][str(i)]['support'] for i in range(10)]}")
    print(f"Total support: {sum(saved_metrics['per_class'][str(i)]['support'] for i in range(10))}")

    # Run inference
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE ON TEST SET")
    print("=" * 70)
    preds = model.predict(X_test, verbose=1)
    pred_labels = np.argmax(preds, axis=1)

    # Per-digit accuracy
    per_digit_acc = np.mean(pred_labels == y_test)
    print(f"\nPer-digit accuracy: {per_digit_acc:.4f} ({per_digit_acc*100:.2f}%)")

    # Full 5-digit exact-match accuracy
    num_groups = test_group_ids.max() + 1
    exact_matches = 0
    for g in range(num_groups):
        idx = np.where(test_group_ids == g)[0]
        if len(idx) == NUM_DIGITS:
            if np.array_equal(pred_labels[idx], y_test[idx]):
                exact_matches += 1
    exact_match_acc = exact_matches / num_groups
    print(f"Full 5-digit exact-match accuracy: {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)")
    print(f"Exact matches: {exact_matches} / {num_groups}")

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Per-digit)")
    print("=" * 70)
    report = classification_report(y_test, pred_labels, digits=4)
    print(report)

    # Confusion matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Counts)")
    print("=" * 70)
    cm = confusion_matrix(y_test, pred_labels, labels=list(range(10)))
    print(cm)
    print(f"\nTotal samples: {cm.sum()} (expected 1,060)")

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Normalized / Recall per class)")
    print("=" * 70)
    for i in range(10):
        print(f"Class {i}: {cm_norm[i]}")

    # Verify against saved metrics
    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST SAVED METRICS")
    print("=" * 70)
    print(f"Per-digit accuracy match: {abs(per_digit_acc - saved_metrics['per_digit_accuracy']) < 0.001}")
    print(f"  Computed: {per_digit_acc:.6f}")
    print(f"  Saved:    {saved_metrics['per_digit_accuracy']:.6f}")
    print(f"  Diff:     {abs(per_digit_acc - saved_metrics['per_digit_accuracy']):.6f}")

    print(f"\nFull 5-digit exact-match match: {abs(exact_match_acc - saved_metrics['full_5digit_exact_match_accuracy']) < 0.001}")
    print(f"  Computed: {exact_match_acc:.6f}")
    print(f"  Saved:    {saved_metrics['full_5digit_exact_match_accuracy']:.6f}")
    print(f"  Diff:     {abs(exact_match_acc - saved_metrics['full_5digit_exact_match_accuracy']):.6f}")

    # Per-class verification
    print("\nPer-class verification:")
    for i in range(10):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = np.mean(pred_labels[class_mask] == i)
            saved_acc = saved_metrics['per_class'][str(i)]['recall']  # recall = per-class accuracy
            match = abs(class_acc - saved_acc) < 0.01
            print(f"  Class {i}: computed={class_acc:.4f}, saved={saved_acc:.4f}, match={match}, support={class_mask.sum()}")

    # Plot confusion matrix
    print("\n" + "=" * 70)
    print("PLOTTING CONFUSION MATRIX")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(10), yticklabels=range(10))
    axes[0].set_title('Confusion Matrix (Counts)\nTest Set: 212 images × 5 digits = 1,060 samples')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=range(10), yticklabels=range(10), vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized / Recall)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "runs", "eval", "exp", "confusion_matrix_recomputed.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Train images: 1,270 → {1270 * 5} = 6,350 digit crops")
    print(f"Val images:   173   → {173 * 5} = 865 digit crops")
    print(f"Test images:  212   → {212 * 5} = 1,060 digit crops")
    print(f"\nTest set size: 212 images (NOT 180 as originally claimed in paper)")
    print(f"Confusion matrix computed on: {cm.sum()} digit instances")
    print(f"Full-sequence accuracy computed on: {num_groups} images")
    print(f"\nPaper claims (FIXED):")
    print(f"  - Dataset split: 1,200 / 220 / 212")
    print(f"  - CNN per-digit accuracy: 76.32%")
    print(f"  - CNN full-sequence exact-match: 56.60% (120/212)")
    print(f"  - PaddleOCR full-sequence exact-match: 94.71% (197/208 same-length)")

if __name__ == "__main__":
    main()