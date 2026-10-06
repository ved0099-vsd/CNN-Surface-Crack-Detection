# Industrial Surface Crack Detection using CNN

## Project Overview

This project implements an **Industrial Surface Crack Detection system using a Convolutional Neural Network (CNN)**.

The model is trained to classify surface images into two categories:

* **Crack**
* **No Crack**

The project uses **TensorFlow/Keras** for building and training the CNN model.

## Features

* Dataset validation
* Automatic Train / Validation / Test split
* Image preprocessing and augmentation
* CNN-based image classification
* Batch Normalization
* Dropout for reducing overfitting
* Early Stopping
* Model Checkpointing
* Learning Rate Reduction
* Training and validation accuracy graphs
* Training and validation loss graphs
* Test-set evaluation
* Confusion Matrix
* Classification Report
* Single-image crack prediction

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

## Dataset

The dataset used for this project was provided as part of the classroom/project training.

The original dataset contains two classes:

```text
CrackDataset/
├── Positive/
└── Negative/
```

The program automatically creates the following structure:

```text
Processed_CrackDataset/
├── train/
│   ├── Crack/
│   └── NoCrack/
├── validation/
│   ├── Crack/
│   └── NoCrack/
└── test/
    ├── Crack/
    └── NoCrack/
```

The dataset is **not included in this public repository**.

## CNN Model

The CNN contains multiple convolutional layers followed by
