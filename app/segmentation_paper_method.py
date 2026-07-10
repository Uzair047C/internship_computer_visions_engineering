"""
Digit segmentation following Son et al., "Deep Learning-based Number
Detection and Recognition for Gas Meter Reading" (IEIE TSPC, 2019).

The paper uses TWO different segmentation methods depending on region,
because connected components alone breaks down when digits sit close
together next to other clutter (their ID region, next to a barcode):

  - Meter-reading area -> connected component labeling, with an
    "exception processing / box middle size" filter and a final
    "rescale" step (paper Fig. 4).
  - ID area            -> MSER (maximally stable extremal regions),
    with height-histogram exception handling, barcode removal by
    width, and NMS to collapse duplicate MSER detections (paper Fig. 5).

    

Drop this in place of the segmentation functions in your existing script.
`segment_digit_boxes_reading_area` is the direct replacement for your
`segment_digit_boxes`. `segment_digit_boxes_id_area` is provided too, in
case you add a second YOLO class for an ID region later -- connected
components will NOT work well there per the paper, MSER is the reason it
exists as a separate method.
"""

import cv2
import numpy as np

# ========================= CONFIG =========================
NUM_DIGITS = 5
DIGIT_HEIGHT = 32
DIGIT_WIDTH = 32
BOX_PADDING_PX = 2

MIN_COMPONENT_AREA_FRAC = 0.01
MAX_COMPONENT_AREA_FRAC = 0.60

# "box middle size" tolerance band -- how far a box's height/width is
# allowed to deviate from the median before it's treated as an exception
# (noise, glare, a partial/broken stroke) rather than a real digit.
HEIGHT_TOL_LOW = 0.5
HEIGHT_TOL_HIGH = 1.6
WIDTH_TOL_LOW = 0.3
WIDTH_TOL_HIGH = 1.8

MERGE_GAP_FRAC = 0.15      # merge components closer than this * median_width
SPLIT_WIDTH_FACTOR = 1.6   # split components wider than this * median_width


# =========================================================================
# METER-READING AREA -- connected component method (paper section 2.2,
# paragraph 1; pipeline shown in Fig. 4 a->b->c->d)
# =========================================================================

def preprocess_reading_area(gray_img, use_adaptive=False, use_clahe=True):
    """(a)->(b) in Fig. 4: filtering + binarization.

    Paper: 'binarization is performed after a filtering process, such as
    bilateral and morphology.' Added CLAHE and an optional adaptive
    threshold here -- the paper's own conclusion flags lighting as their
    unsolved failure mode, so a single global Otsu threshold is the most
    likely place your 40% failures are coming from if they cluster around
    glare/shadow rather than touching digits.
    """
    img = gray_img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    if use_adaptive:
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=25, C=7
        )
    else:
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(binary == 255) > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def connected_component_boxes(binary_img):
    """(c) in Fig. 4: connected component labeling."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    boxes = []
    img_area = binary_img.shape[0] * binary_img.shape[1]
    for label_id in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[label_id]
        area_frac = area / img_area
        if MIN_COMPONENT_AREA_FRAC <= area_frac <= MAX_COMPONENT_AREA_FRAC:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def exception_process_by_box_middle_size(boxes):
    """The paper's 'exception processing and box middle size are
    referenced' step. All real digits in one reading-area crop share the
    same font/baseline, so they cluster tightly around a median height and
    width. Anything well outside that band -- a screw, a reflection, half
    a broken stroke -- is dropped here even if its raw area happened to
    pass the area-fraction filter above.
    """
    if len(boxes) == 0:
        return boxes
    median_h = np.median([b[3] for b in boxes])
    return [b for b in boxes if HEIGHT_TOL_LOW * median_h <= b[3] <= HEIGHT_TOL_HIGH * median_h]


def merge_fragmented_boxes(boxes):
    """Not explicit in the paper's text, but necessary in practice: a
    single digit whose stroke was split into two components by glare or a
    thin joint that thresholding erased. Merge components that sit close
    together horizontally into one box before deciding digit count --
    otherwise the 'keep top N by size' step below will happily discard one
    half of a real digit as noise.
    """
    if len(boxes) <= 1:
        return boxes
    median_w = np.median([b[2] for b in boxes])
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:
        px, py, pw, ph = merged[-1]
        x, y, w, h = b
        gap = x - (px + pw)
        if gap < MERGE_GAP_FRAC * median_w:
            nx0, ny0 = min(px, x), min(py, y)
            nx1, ny1 = max(px + pw, x + w), max(py + ph, y + h)
            merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        else:
            merged.append(b)
    return merged


def split_merged_box(binary_crop, expected_splits):
    """Ink-density valley split for touching/merged digits. Not detailed
    in the paper (their reading-area digits are spaced enough that this
    mostly doesn't come up for them), but standard classical-CV technique
    for the failure case you described where components merge.
    """
    col_sums = binary_crop.sum(axis=0)
    w = binary_crop.shape[1]
    seg_width = max(1, w // expected_splits)

    split_points = [0]
    for i in range(1, expected_splits):
        approx = i * seg_width
        lo = max(0, approx - seg_width // 3)
        hi = min(w, approx + seg_width // 3)
        window = col_sums[lo:hi]
        if len(window) == 0:
            split_points.append(approx)
            continue
        split_points.append(lo + int(np.argmin(window)))
    split_points.append(w)

    split_points = sorted(set(split_points))
    if len(split_points) < expected_splits + 1:
        split_points = [int(i * w / expected_splits) for i in range(expected_splits + 1)]

    return [(split_points[i], split_points[i + 1]) for i in range(len(split_points) - 1)]


def split_oversized_boxes(boxes, binary_img, num_digits):
    if len(boxes) == 0:
        return boxes
    median_w = np.median([b[2] for b in boxes])
    new_boxes = []
    for (x, y, w, h) in boxes:
        if median_w > 0 and w > median_w * SPLIT_WIDTH_FACTOR:
            n_merged = min(max(2, round(w / median_w)), num_digits)
            sub_crop = binary_img[y:y + h, x:x + w]
            for (sx0, sx1) in split_merged_box(sub_crop, n_merged):
                new_boxes.append((x + sx0, y, sx1 - sx0, h))
        else:
            new_boxes.append((x, y, w, h))
    new_boxes.sort(key=lambda b: b[0])
    return new_boxes


def rescale_to_common_band(boxes, img_shape):
    """(d) in Fig. 4: 'a rescale process is performed to detect the final
    number area.' All digits in one reading window sit on the same
    baseline, so snap every surviving box's vertical extent to the union
    band of all boxes -- this recovers a digit that lost its top/bottom
    pixel row to thresholding instead of feeding the CNN a cropped digit.
    """
    if len(boxes) == 0:
        return boxes
    y0 = max(0, min(b[1] for b in boxes) - 1)
    y1 = min(img_shape[0], max(b[1] + b[3] for b in boxes) + 1)
    return [(x, y0, w, y1 - y0) for (x, y, w, h) in boxes]


def segment_digit_boxes_reading_area(gray_img, num_digits=NUM_DIGITS, use_adaptive=False):
    """Full pipeline matching paper Fig. 4 (a)->(b)->(c)->(d).
    Drop-in replacement for your original `segment_digit_boxes`.
    """
    h_img, w_img = gray_img.shape[:2]
    binary = preprocess_reading_area(gray_img, use_adaptive=use_adaptive)

    boxes = connected_component_boxes(binary)
    boxes = exception_process_by_box_middle_size(boxes)
    boxes = merge_fragmented_boxes(boxes)
    boxes = split_oversized_boxes(boxes, binary, num_digits)
    boxes = exception_process_by_box_middle_size(boxes)  # re-check post merge/split
    boxes = rescale_to_common_band(boxes, gray_img.shape)

    if len(boxes) == num_digits:
        return boxes

    if len(boxes) > num_digits:
        # Keep the num_digits boxes closest to median shape, not just the
        # largest by area -- favors "digit-shaped" over "big blob".
        median_w = np.median([b[2] for b in boxes])
        median_h = np.median([b[3] for b in boxes])

        def shape_score(b):
            return abs(b[2] - median_w) + abs(b[3] - median_h)

        boxes = sorted(boxes, key=shape_score)[:num_digits]
        boxes.sort(key=lambda b: b[0])
        return boxes

    # Last-resort fallback only -- equal-width slicing throws away all
    # the segmentation work above, so treat a hit here as a signal to go
    # look at that image with debug_show_segmentation_split().
    col_w = w_img // num_digits
    return [(i * col_w, 0, col_w, h_img) for i in range(num_digits)]


# =========================================================================
# ID AREA -- MSER method (paper section 2.2, paragraph 2; Fig. 5 a->b->c->d)
# Use this if you add a second YOLO-detected region for an ID/serial
# number block where digits sit close together next to a barcode.
# Connected components struggles there per the paper; this is why MSER
# exists as a second method rather than reusing the reading-area pipeline.
# =========================================================================

def preprocess_id_area(gray_img):
    """Paper: 'a color reversal process is added, because the color is
    opposite to that of the meter-reading area, and binarization does not
    proceed.' MSER runs directly on the (inverted) grayscale, no threshold.
    """
    return cv2.bitwise_not(gray_img)


def mser_boxes(gray_img, delta=5, min_area=30, max_area_frac=0.25):
    """(b)->(c) in Fig. 5: MSER region detection."""
    img = preprocess_id_area(gray_img)
    h_img, w_img = img.shape[:2]
    mser = cv2.MSER_create(delta=delta, min_area=min_area,
                            max_area=int(max_area_frac * h_img * w_img))
    regions, _ = mser.detectRegions(img)
    return [cv2.boundingRect(r.reshape(-1, 1, 2)) for r in regions]


def remove_barcode_by_width(boxes):
    """Paper: 'the bar code is removed by filtering.' Barcode bars are
    much narrower/taller than digit glyphs (high aspect ratio); digits are
    roughly square-ish. Filter on aspect ratio to drop bar-shaped regions.
    """
    filtered = []
    for (x, y, w, h) in boxes:
        if h == 0:
            continue
        aspect = w / float(h)
        if 0.25 <= aspect <= 1.3:
            filtered.append((x, y, w, h))
    return filtered


def exception_by_height_histogram(boxes):
    """Paper: 'the max value of the smoothed histogram by height is
    referred to as an exception, and the width is also processed.'
    Approximated here with a median-height + median-width band, same idea
    as the reading-area's box-middle-size filter above.
    """
    if len(boxes) == 0:
        return boxes
    med_h = np.median([b[3] for b in boxes])
    med_w = np.median([b[2] for b in boxes])
    return [b for b in boxes
            if HEIGHT_TOL_LOW * med_h <= b[3] <= HEIGHT_TOL_HIGH * med_h
            and WIDTH_TOL_LOW * med_w <= b[2] <= WIDTH_TOL_HIGH * med_w]


def nms_boxes(boxes, iou_thresh=0.3):
    """(d) in Fig. 5: 'the final box is a duplicate form, which uses
    non-maximum suppression (NMS) to combine the overlapping regions into
    a single image.' MSER produces many overlapping candidate regions per
    real digit; NMS collapses them to one box per digit.
    """
    if len(boxes) == 0:
        return boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=np.float32)
    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(rects[i, 0], rects[order[1:], 0])
        yy1 = np.maximum(rects[i, 1], rects[order[1:], 1])
        xx2 = np.minimum(rects[i, 2], rects[order[1:], 2])
        yy2 = np.minimum(rects[i, 3], rects[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thresh]
    kept = [boxes[i] for i in keep]
    kept.sort(key=lambda b: b[0])
    return kept


def segment_digit_boxes_id_area(gray_img, num_digits):
    """Full pipeline matching paper Fig. 5 (a)->(b)->(c)->(d)."""
    h_img, w_img = gray_img.shape[:2]

    boxes = mser_boxes(gray_img)
    boxes = remove_barcode_by_width(boxes)
    boxes = exception_by_height_histogram(boxes)
    boxes = nms_boxes(boxes)

    if len(boxes) > num_digits:
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        boxes = boxes[:num_digits]
        boxes.sort(key=lambda b: b[0])

    if len(boxes) == num_digits:
        return boxes

    col_w = w_img // num_digits
    return [(i * col_w, 0, col_w, h_img) for i in range(num_digits)]


# =========================================================================
# Shared crop extraction (unchanged from your original -- works with boxes
# from either pipeline above)
# =========================================================================

def extract_digit_crops(gray_img, boxes, target_size=(DIGIT_HEIGHT, DIGIT_WIDTH)):
    h_img, w_img = gray_img.shape[:2]
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


# =========================================================================
# Debug visualizer -- same idea as your original, but shows every stage
# (raw components -> after exceptions -> after merge -> after split) so
# you can see exactly which stage is discarding/misplacing a digit on
# your failing images.
# =========================================================================

def debug_show_segmentation_stages(image_path, num_digits=NUM_DIGITS, use_adaptive=False):
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise FileNotFoundError(f"Could not load image:\n{image_path}")

    binary = preprocess_reading_area(gray_img, use_adaptive=use_adaptive)
    raw = connected_component_boxes(binary)
    after_exc = exception_process_by_box_middle_size(raw)
    after_merge = merge_fragmented_boxes(after_exc)
    after_split = split_oversized_boxes(after_merge, binary, num_digits)
    final_boxes = segment_digit_boxes_reading_area(gray_img, num_digits, use_adaptive)

    stages = [("raw components", raw), ("after exceptions", after_exc),
              ("after merge", after_merge), ("after split", after_split),
              ("final", final_boxes)]

    fig, axes = plt.subplots(1, len(stages), figsize=(3.5 * len(stages), 4))
    for ax, (title, boxes) in zip(axes, stages):
        annotated = Image.fromarray(gray_img).convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for (x, y, w, h) in boxes:
            draw.rectangle([(x, y), (x + w, y + h)], outline="lime", width=1)
        ax.imshow(annotated)
        ax.set_title(f"{title} ({len(boxes)})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return final_boxes
