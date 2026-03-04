# ContextFire-VLM

A research project focused on context-aware fire detection using Vision-Language Models (VLMs). This repository contains the full pipeline from data preprocessing and augmentation to model benchmarking and fine-tuning.

## Directory Structure

### Notebooks

* **data_preprocessing_notebooks/**
  * Steps for data preparation including format conversion, image augmentation, automated labeling, data cleaning, and exploratory data analysis (EDA).
* **model_survey_notebooks/**
  * Benchmarking and zero-shot evaluation of various VLMs including Gemma 3, SmolVLM, InternVL, and Qwen-VL variants.
* **vlm_finetune/**
  * Implementation of fine-tuning pipelines for InternVL3 and Qwen2.5-VL, including performance evaluation and training visualization.

### Data and Scripts

* **fine_tune_dataset/**: Structured images and labels organized into train, validation, and test splits for model training.
* **dataset_v2/** and **new_images/**: Repositories for raw and processed image assets used across the project.
* **scripts/**: Utility scripts, including `survey_model.py` for unified model evaluation across different backends.

### Core Files

* **main.py**: Script for loading and merging fine-tuned model weights.
* **requirements.txt**: Project dependencies.
* ***.csv**: Various label files including `FIRENET.csv`, `SYN_FIRE.csv`, and `labels_v2.csv`.
