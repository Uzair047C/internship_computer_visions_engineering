# Training a Custom Digit-Only PaddleOCR Recognition Model

**Goal:** Fine-tune a PaddleOCR text-recognition model on your own 5-digit meter-strip
images (image path + label in a CSV) so it reads **only characters `0-9`** — no
letters, no symbols — and outputs a 5-digit string per image.

**Scope note:** this workflow is for **recognition only** (`rec`), not detection
(`det`). It assumes your images are already cropped to the digit strip (like your
existing `TRAIN_IMG_DIR`/`train.csv` setup) — i.e. one image = one 5-digit reading,
no need to *find* the digits in a larger scene. If your images are full meter photos
where the digit strip still needs to be located first, you need a `det` step (or
your existing YOLO ROI detector) before this — don't skip that, PaddleOCR's `rec`
models assume the crop is already tight around the text.

---

## 0. Prerequisites

- Windows machine (matches your existing setup at `D:\Meter reading\...`)
- Python 3.8–3.11 (PaddleOCR/PaddlePaddle do **not** yet reliably support 3.13,
  which is what your VGG notebook is running under — check this first, it's the
  most common source of install failures)
- GPU strongly recommended but not required (recognition models are small; CPU
  training is workable for a dataset this size, just slower)
- ~2 GB free disk for pretrained weights + PaddleOCR repo

Check your Python version before doing anything else:

```powershell
python --version
```

If it's 3.13 (like your VGG env), create a **separate** virtual environment for
this — don't try to force PaddlePaddle into the same env as your Keras/TF project.

```powershell
python -m venv paddle_env
paddle_env\Scripts\activate
python -m pip install --upgrade pip
```

---

## 1. Install PaddlePaddle + PaddleOCR

```powershell
# CPU version (swap for the CUDA wheel from paddlepaddle.org.cn if you have a GPU)
pip install paddlepaddle==2.6.1

# Clone PaddleOCR (training tools live in the repo, not just the pip package)
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
pip install -e .
```

**Sanity check** (do this before touching your dataset — catches broken installs early):

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

You should see `PaddlePaddle is installed successfully!`. If this fails, fix it
before proceeding — everything below depends on it.

> Note on your earlier PaddleOCR history: your project notes mention a
> PaddlePaddle 3.3.0 oneDNN regression that needed `enable_mkldnn=False`. I'm
> pinning `2.6.1` above specifically to sidestep that — if you need a newer
> version for another reason, keep `enable_mkldnn=False` in mind as a known
> workaround if you hit crashes on grayscale/small inputs again.

---

## 2. Dataset Preparation

### 2.1 Expected folder layout

```
paddle_digit_project/
├── images/
│   ├── train/        <- copy your existing train images here
│   ├── valid/
│   └── test/
├── labels/
│   ├── train_list.txt
│   ├── val_list.txt
│   └── test_list.txt
└── digit_dict.txt
```

### 2.2 PaddleOCR label file format

PaddleOCR recognition training expects a plain text file, **one sample per line**,
tab-separated:

```
images/train/00017.png	00017
images/train/04213.png	04213
```

- Path is relative to a `--label_file_list` root you'll set in the config.
- Label is the literal ground-truth string — no leading/trailing spaces.
- Because your readings are fixed 5-digit strings, every label here should be
  exactly 5 characters, zero-padded (matching what you already do in your VGG
  notebook with `.zfill(num_digits)`).

### 2.3 Conversion script: your CSV → PaddleOCR label files

This assumes your CSVs have an `image` column (filename) and a `label` column
(the meter reading as an integer), matching your existing `train.csv` /
`valid.csv` / `test.csv` structure. **Adjust column names if yours differ.**

```python
# csv_to_paddle_labels.py
import os
import shutil
import pandas as pd

NUM_DIGITS = 5  # matches your existing pipeline

def convert_split(csv_path, src_img_dir, dst_img_dir, out_label_path, num_digits=NUM_DIGITS):
    """Reads your existing CSV (image, label) and writes:
       1. A PaddleOCR label file: <relative_img_path>\t<zero_padded_label>
       2. Copies images into dst_img_dir (skip if already there)."""
    df = pd.read_csv(csv_path)
    os.makedirs(dst_img_dir, exist_ok=True)

    written, skipped = 0, 0
    with open(out_label_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            img_name = str(row["image"])
            src_path = os.path.join(src_img_dir, img_name)
            if not os.path.exists(src_path):
                skipped += 1
                continue

            dst_path = os.path.join(dst_img_dir, img_name)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

            label_str = str(int(row["label"])).zfill(num_digits)
            if len(label_str) != num_digits:
                # label longer than expected digits -- flag rather than silently truncate
                print(f"WARNING: {img_name} label '{label_str}' is not {num_digits} digits, skipping.")
                skipped += 1
                continue

            # PaddleOCR label files use forward-slash relative paths
            rel_path = os.path.join(os.path.basename(dst_img_dir), img_name).replace("\\", "/")
            f.write(f"{rel_path}\t{label_str}\n")
            written += 1

    print(f"[{os.path.basename(out_label_path)}] wrote {written} lines, skipped {skipped}.")


if __name__ == "__main__":
    PROJECT_ROOT = "paddle_digit_project"
    os.makedirs(os.path.join(PROJECT_ROOT, "labels"), exist_ok=True)

    # ---- EDIT THESE to match your actual existing paths ----
    convert_split(
        csv_path="../data_set_generator_for_cnns_only/data_set/train.csv",
        src_img_dir="../data_set_generator_for_cnns_only/data_set/train",
        dst_img_dir=os.path.join(PROJECT_ROOT, "images", "train"),
        out_label_path=os.path.join(PROJECT_ROOT, "labels", "train_list.txt"),
    )
    convert_split(
        csv_path="../data_set_generator_for_cnns_only/data_set/valid.csv",
        src_img_dir="../data_set_generator_for_cnns_only/data_set/valid",
        dst_img_dir=os.path.join(PROJECT_ROOT, "images", "valid"),
        out_label_path=os.path.join(PROJECT_ROOT, "labels", "val_list.txt"),
    )
    convert_split(
        csv_path="../data_set_generator_for_cnns_only/data_set/test.csv",
        src_img_dir="../data_set_generator_for_cnns_only/data_set/test",
        dst_img_dir=os.path.join(PROJECT_ROOT, "images", "test"),
        out_label_path=os.path.join(PROJECT_ROOT, "labels", "test_list.txt"),
    )
```

Run it:

```powershell
python csv_to_paddle_labels.py
```

---

## 3. The Digit-Only Dictionary (this is what actually restricts output to 0-9)

This is the core mechanism for your "only output 0-9" requirement. PaddleOCR's
recognition head is a classifier over **whatever characters are in this
dictionary file** (plus a CTC blank token it manages internally) — it is
architecturally incapable of predicting a character that isn't in this file.
You don't need any extra output-filtering logic; the restriction happens at the
model-definition level.

Create `paddle_digit_project/digit_dict.txt` with **exactly one character per
line, in this order**:

```
0
1
2
3
4
5
6
7
8
9
```

No blank lines, no extra characters, no header. Ten lines, ten digits. This
file gets referenced in the config below as `character_dict_path`, and it
determines the size of the model's final classification layer — training will
build a 10-class (+blank) output head from this, not the full ~6600-character
dictionary PaddleOCR ships with by default for general Chinese+English OCR.

---

## 4. Model Config

Use PP-OCRv4's English recognition config as the base — it's small, fast, and
its default alphabet is already Latin/digit-oriented, so less needs to change
than if you started from the Chinese-focused default.

Copy the base config so you don't edit PaddleOCR's shipped version:

```powershell
copy configs\rec\PP-OCRv4\en_PP-OCRv4_rec.yml configs\rec\PP-OCRv4\digit_only_rec.yml
```

Open `configs/rec/PP-OCRv4/digit_only_rec.yml` and replace it with this
(full file, not a diff — some fields interact, easier to replace wholesale):

```yaml
Global:
  debug: false
  use_gpu: true              # set false if you don't have a CUDA GPU
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/digit_only_rec/
  save_epoch_step: 5
  eval_batch_step: [0, 200]
  cal_metric_during_train: true
  pretrained_model: ./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/en/word_1.png
  character_dict_path: ./paddle_digit_project/digit_dict.txt
  max_text_length: 5         # your readings are always 5 digits -- fixed length
  infer_mode: false
  use_space_char: false      # no space character allowed in output
  distributed: false
  save_res_path: ./output/digit_only_rec/predicts.txt

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005    # lower than default since we're fine-tuning, not training from scratch
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: true
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: 5

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./paddle_digit_project/images/train/
    ext_op_transform_idx: 1
    label_file_list:
      - ./paddle_digit_project/labels/train_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecConAug:
          prob: 0.5
          ext_data_num: 2
          image_shape: [48, 320, 3]
          max_text_length: 5
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./paddle_digit_project/images/valid/
    label_file_list:
      - ./paddle_digit_project/labels/val_list.txt
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
```

A few things worth being explicit about since I'm not going to hide the
trade-offs of this config:

- `RecAug` is PaddleOCR's built-in recognition augmentation (adds blur, noise,
  jitter, color changes). It does **not** include large rotations by default —
  it's designed for natural scene text, so it's already reasonably mild on the
  geometric side. If you want to be extra safe about not introducing rotation
  given your existing deskew step, you can disable it and rely on `RecConAug`
  alone, or write a custom transform — say so if you want that version instead
  of PaddleOCR's default augmentation stack.
- `max_text_length: 5` hard-caps output length. If any of your labels are
  shorter/longer than 5 digits after zero-padding, training will error or
  silently misalign — the conversion script above already flags mismatches.
- I'm using the SVTR_LCNet/PP-OCRv4 architecture as-is rather than shrinking it
  further. Given your dictionary is only 10 classes, you *could* get away with
  a much smaller custom backbone, but reusing PaddleOCR's stock architecture
  means you keep compatibility with their pretrained weights (below) and their
  export/inference tooling, which is worth more than the parameter savings at
  your dataset size.

---

## 5. Download Pretrained Weights (for fine-tuning, not training from scratch)

```powershell
mkdir pretrain_models
cd pretrain_models
# PP-OCRv4 English recognition pretrained weights
curl -O https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar
tar -xf en_PP-OCRv4_rec_train.tar
cd ..
```

This matches the `pretrained_model` path already set in the config above. Even
though your final task only needs 10 classes, starting from a model that
already knows general-purpose stroke/edge features for Latin characters gives
you a real head start over random initialization, especially with a training
set in the low thousands.

---

## 6. Train

```powershell
python tools/train.py -c configs/rec/PP-OCRv4/digit_only_rec.yml
```

Watch for:
- `acc` metric printed at each eval step — this is exact-string-match accuracy
  (all 5 digits correct), analogous to your VGG pipeline's
  "Full 5-digit exact-match accuracy".
- If loss goes to `nan` early on, your learning rate is too high for
  fine-tuning — drop `Optimizer.lr.learning_rate` to `0.0001` and retry.

To resume from a checkpoint after an interruption:

```powershell
python tools/train.py -c configs/rec/PP-OCRv4/digit_only_rec.yml -o Global.checkpoints=./output/digit_only_rec/latest
```

---

## 7. Evaluate

```powershell
python tools/eval.py -c configs/rec/PP-OCRv4/digit_only_rec.yml -o Global.pretrained_model=./output/digit_only_rec/best_accuracy
```

For a held-out test set (not just the validation split used during training),
point `Eval.dataset.data_dir` and `label_file_list` in a copy of the config at
your `test_list.txt` instead, and re-run `tools/eval.py` against that copy —
same pattern you already use for `X_test` in your VGG notebook.

---

## 8. Export for Inference

Training checkpoints aren't directly usable for fast inference — export to
Paddle's inference format:

```powershell
python tools/export_model.py -c configs/rec/PP-OCRv4/digit_only_rec.yml -o Global.pretrained_model=./output/digit_only_rec/best_accuracy Global.save_inference_dir=./inference/digit_only_rec/
```

---

## 9. Run Inference (Python, restricted to digit output)

```python
# infer_digit_meter.py
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=False,       # no rotation-correction stage -- you already deskew upstream
    det=False,                 # recognition only -- your images are already cropped to the digit strip
    rec_model_dir="./inference/digit_only_rec/",
    rec_char_dict_path="./paddle_digit_project/digit_dict.txt",
    rec_image_shape="3,48,320",
    use_gpu=False,             # set True if available
)

def read_meter_paddle(image_path):
    result = ocr.ocr(image_path, det=False, cls=False)
    # result format: [[(text, confidence)]]
    text, confidence = result[0][0]
    return text, confidence

if __name__ == "__main__":
    reading, conf = read_meter_paddle("paddle_digit_project/images/test/00017.png")
    print(f"Reading: {reading}  (confidence: {conf*100:.2f}%)")
```

Because `rec_char_dict_path` points at your 10-line digit-only dictionary, the
model is *structurally* unable to output anything other than `0-9` — there is
no separate "filter out non-digits" step needed, and no risk of a stray letter
slipping through post-hoc filtering.

---

## 10. Where this fits next to your existing VGG pipeline

You now have two independent digit-reading approaches on the same data:
your VGG+segmentation classifier, and this PaddleOCR recognition model. I'd
suggest evaluating both on the **same held-out test set** with the **same
exact-match metric** (`evaluate_full()`'s `exact_match_acc` vs. this model's
`acc`) before deciding which one feeds `amr_pipeline.py` — don't assume the
newer one is better just because it's newer. Given your dataset size (low
thousands of images), it's genuinely uncertain which will generalize better,
and that's worth settling with numbers rather than intuition.

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `paddle.utils.run_check()` fails | Wrong Paddle/Python version pairing | Recheck Python version, reinstall matching Paddle wheel |
| Training loss is `nan` from step 1 | LR too high for fine-tuning | Lower `Optimizer.lr.learning_rate` to `1e-4` |
| All predictions come out as blank/empty string | `character_dict_path` mismatch between train config and inference config | Confirm both point at the exact same `digit_dict.txt` |
| Accuracy stuck near 0% even after many epochs | Label file paths don't resolve (silently loads 0 valid samples) | Check `SimpleDataSet` `data_dir` is correct relative to where you launch `tools/train.py` from |
| Crashes on grayscale images specifically | oneDNN/mkldnn regression (seen previously in this project on PaddlePaddle 3.3.0) | Pin PaddlePaddle to `2.6.1` as above, or pass `enable_mkldnn=False` |
