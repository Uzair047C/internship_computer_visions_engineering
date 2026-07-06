# Internship Computer Vision Engineering

This repository contains experiments and pipelines for meter/object detection, OCR, dataset generation, and model training.

## Project layout

- `app/`: application notebooks and utilities.
- `coordinates_function/`: coordinate extraction notebook and tests.
- `data_Set/`: YOLO-style dataset folders and YAML config.
- `data_set_generator_for_cnns_only/`: dataset generation workflow for CNN training.
- `model_training/`: model training notebooks, DVC pipeline, and inference runs.
- `cnns_training/`: saved CNN model artifacts.
- `ocr_function/`: OCR notebooks and output samples.
- `research_paper_performance_improvement/`: research notes and experiments.
- `structure_and_performance/`: architecture and performance documentation.

## Notes

- Large datasets and generated artifacts should be managed with DVC/storage and are ignored by default in Git.
- Jupyter notebooks are tracked as binary in merge operations to reduce corruption/conflict risk.
