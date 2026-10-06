# Industrial Surface Crack Detection using CNN

## Project Overview

This project implements an **Industrial Surface Crack Detection system using a Convolutional Neural Network (CNN)**.

The model classifies industrial surface images into two categories:

* **Crack**
* **No Crack**

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

## Virtual Environment Setup

This project uses a **Python virtual environment** to manage the required libraries and keep the project dependencies separate from the system Python installation.

### 1. Create a Virtual Environment

Open the terminal inside the project folder and run:

```bash
python -m venv venv
```

This creates a new folder named `venv` containing the virtual environment.

### 2. Activate the Virtual Environment

For **Windows**:

```bash
venv\Scripts\activate
```

After activation, the terminal will show `(venv)` before the current path.

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Deactivate the Virtual Environment

When finished working on the project:

```bash
deactivate
```

## Dataset

The project requires the **Crack Dataset** provided for the project.

The original dataset should have the following structure:

```text
CrackDataset/
├── Positive/
└── Negative/
```

* `Positive` → Crack images
* `Negative` → No Crack images

The program automatically creates a processed dataset:

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

The dataset is divided approximately into:

```text
70% → Training
15% → Validation
15% → Testing
```

## CNN Model

The CNN model contains multiple convolutional layers with:

* Batch Normalization
* Max Pooling
* Flatten
* Dense Layers
* Dropout
* Sigmoid Output Layer

The model performs binary classification between **Crack** and **No Crack**.

## Training Configuration

```text
Image Size : 128 × 128
Batch Size : 32
Epochs     : 15
```

## Model Training

The project uses:

* Adam Optimizer
* Binary Cross Entropy Loss
* Accuracy
* Early Stopping
* Model Checkpoint
* Reduce Learning Rate on Plateau

## Model Evaluation

The trained model is evaluated using:

* Test Accuracy
* Test Loss
* Confusion Matrix
* Classification Report

The project also generates training and validation accuracy/loss graphs.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/ved0099-vsd/CNN-Crack-Detection.git
```

### 2. Open the Project Folder

```bash
cd CNN-Crack-Detection
```

### 3. Create the Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the Crack Dataset

Place the dataset inside the project folder:

```text
CNN-Crack-Detection/
├── CrackDataset/
│   ├── Positive/
│   └── Negative/
├── Crack_Detection.py
├── README.md
└── requirements.txt
```

### 7. Run the Program

```bash
python Crack_Detection.py
```

The program will create the processed dataset, train the CNN model, evaluate the model, and perform single-image prediction.

## Author

**Vedant Dhamal**

## Project

**Industrial Surface Crack Detection using Convolutional Neural Network (CNN)**

