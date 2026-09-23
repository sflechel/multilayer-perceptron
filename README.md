# Custom Multilayer Perceptron (MLP) from Scratch

A modular, production-ready Deep Neural Network (DNN) and feature scaling pipeline built entirely from scratch using **NumPy** (no PyTorch, TensorFlow, or high-level machine learning frameworks).

This project was developed, trained, and tested on the **WDBC (Wisconsin Diagnostic Breast Cancer)** dataset to classify whether breast cancer cell nuclei features are malignant or benign.

---

## Project Structure

```text
.
├── clean_and_split_dataset.py   # Dataset preprocessing and train/val splitting
├── training.py                  # Model training loop, early stopping, and history export
├── prediction.py                # Inference script for making patient predictions
├── compare_runs.py              # Visual comparison tool for multiple training history logs
├── data.csv                     # Raw dataset
├── mlp/
│   ├── layer_class.py           # Individual layer implementation (forward/backward passes)
│   └── multilayer_perceptron.py # Network orchestration and optimization routines
├── utils/
│   ├── activation.py            # Activation functions and their derivatives (ReLU, Sigmoid, etc.)
│   ├── feature_scaler.py        # Standardization and normalization scaler logic
│   └── initializers.py          # Weight initialization strategies (He, Xavier, Zero)
├── logs/                        # Saved JSON training history logs for comparison
└── output/                      # Serialized model weights and bundles
```

---

## Features

* **Modular Architecture:** Dynamic network construction supporting arbitrary hidden layer configurations.
* **Optimizers:** Full support for Standard Gradient Descent (`none`), Momentum, RMSprop, and Adam with hyperparameter tuning (`--beta1`, `--beta2`).
* **Initializers & Activations:** Supports He, Xavier, and Zero initializers combined with ReLU, Sigmoid, Tanh, and Linear activation functions.
* **Regularization & Early Stopping:** Built-in L1/L2 regularization and patience-based early stopping with automatic state restoration.
* **Comprehensive Metrics & Logging:** Tracks Loss, Accuracy, Precision and Recall per epoch with hierarchical logging propagation.
* **Run Comparison CLI:** Dedicated visualization tool (`compare_runs.py`) to benchmark multiple training runs simultaneously across 2x2 grid subplots.

---

## Getting Started

### Prerequisites

Make sure you have Python installed along with the required numerical and plotting libraries:

```bash
pip install .
```

---

## Usage Guide

### 1. Preprocess and Split the Dataset

Clean the raw data and split it into training and validation sets:

```bash
python clean_and_split_dataset.py --split 0.8 --seed 42 data.csv
```

### 2. Train the Model

Train the MLP using customizable hyperparameters, activation functions, optimizers, and regularization:

```bash
python training.py \
    --layer 24 24 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    -activ relu \
    -init he \
    -reg l2 \
    -l 0.01 \
    --patience 10 \
    --optimizer adam \
    --seed 42
```

* **Metrics logs** are automatically serialized into JSON files within the `logs/` directory, named dynamically after the run's exact configuration parameters.

### 3. Make Predictions

Run inference on unseen data using trained weights:

```bash
python prediction.py path/to/test_data.csv output/model_weights.pkl
```

* Outputs likelihood estimates for each patient (e.g., *"patient 1 has 98.45% likelihood of having malignant breast cancer"*).

### 4. Compare Multiple Training Runs

Use the comparison tool to overlay performance curves from multiple saved JSON history files:

```bash
python compare_runs.py logs/*
```

* Opens a 2x2 grid comparing Validation Loss, Accuracy, Precision, and Recall across all specified runs with filename-based legend mapping.
