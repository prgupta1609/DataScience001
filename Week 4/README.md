# Week 4: CIFAR-10 Image Classification using ANN and CNN

This project builds an end-to-end image classification system on the CIFAR-10 dataset.

## Objective

The objective is to train and compare different neural network architectures for image classification:

- Artificial Neural Network (ANN)
- Basic Convolutional Neural Network (CNN)
- Improved CNN with better training strategies

## Dataset

The notebook uses the CIFAR-10 dataset from TensorFlow/Keras.

CIFAR-10 contains 60,000 color images of size 32x32 from 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## What is Covered

- Loading CIFAR-10 dataset
- Image visualization
- Data preprocessing and normalization
- ANN baseline model
- Basic CNN model
- Improved CNN model
- Dropout
- Batch Normalization
- Data Augmentation
- EarlyStopping
- ReduceLROnPlateau
- Accuracy/loss curve analysis
- Model comparison
- Confusion matrix
- Class-wise classification report
- Final architecture and training strategy analysis

## Files

- `week4_PriyaGupta.ipynb` - Main notebook
- `requirements_week4.txt` - Required libraries for Week 4
- `README.md` - Project documentation

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook and run all cells.

For best performance, run the notebook on Google Colab with GPU enabled.

## Note

If TensorFlow installation gives issues on local Python, use Google Colab because it already supports TensorFlow and GPU acceleration.

