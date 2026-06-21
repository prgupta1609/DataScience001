# Week 6: Denoising Autoencoder on MNIST

## Objective

The objective of this assignment is to build a Deep Learning model capable of removing noise from handwritten digit images using a Denoising Autoencoder. The model is trained on noisy MNIST images and learns to reconstruct the original clean images.

---

## Dataset

* Dataset: MNIST Handwritten Digits
* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Total Classes: 10 digits (0–9)

---

## Steps Performed

### 1. Data Loading and Preprocessing

* Loaded MNIST dataset.
* Normalized pixel values to the range [0,1].
* Reshaped images for neural network processing.

### 2. Noise Generation

* Added Gaussian noise to training and testing images.
* Clipped pixel values to keep them within valid range.

### 3. Denoising Autoencoder Construction

The autoencoder consists of:

#### Encoder

* Input Layer
* Dense Layer (128 neurons, ReLU)
* Dense Layer (64 neurons, ReLU)

#### Decoder

* Dense Layer (128 neurons, ReLU)
* Output Layer (784 neurons, Sigmoid)

### 4. Model Training

* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Epochs: 20
* Batch Size: 256

### 5. Image Reconstruction

* Noisy test images were provided as input.
* The trained autoencoder generated denoised versions of the images.

---

## Results and Observations

* The model successfully removed a significant amount of noise from handwritten digits.
* Reconstructed images preserved the overall digit structure.
* Some fine details were slightly blurred due to compression in the latent representation.
* Increasing training epochs improved reconstruction quality.
* The autoencoder learned meaningful image representations without using class labels.

---

## Conclusion

A Denoising Autoencoder was successfully implemented on the MNIST dataset. The model learned to reconstruct clean digit images from noisy inputs and demonstrated the effectiveness of unsupervised feature learning. The experiment highlights how autoencoders can be used for image restoration and noise reduction tasks.

---






