An advanced deep learning project that implements a **Fully Convolutional Denoising Autoencoder (DAE)** to eliminate synthetic Gaussian noise from handwritten digits using the MNIST dataset. The network passes corrupted inputs through a tight dimensional bottleneck layer, forcing convolutional kernels to learn robust latent representations rather than an identity mapping.

---

## 📂 Project Structure

```text
DataScience001/
└── Week 6/
    ├── data/
    │   └── mnist_png/
    │       ├── testing/            # 10,000 test images organized into 0-9 subfolders
    │       └── training/           # 60,000 train images organized into 0-9 subfolders
    ├── week6_PriyaGupta.ipynb      
    ├── requirements.txt            
    └── README.md                   

🛠️ Technical Stack & Dependencies
This framework is built natively on Python 3.13. Dependencies are specified below:

Core Engine: tensorflow>=2.10.0

Computer Vision Processing: opencv-python>=4.5.5

Numerical Computing: numpy>=1.22.0

Plotting & Metrics visualization: matplotlib>=3.5.0

🧠 Model Architecture & Pipeline Design
The system implements a perfectly symmetric encoder-decoder feature matching map:1. The Encoder NetworkTranslates structural inputs downwards to filter out random ambient signal variances:Input Layer: (None, 28, 28, 1) raw normalized pixel array.Feature Maps: Two consecutive blocks of Conv2D layers ($32$ filters, $3\times3$ kernels, ReLU activation) coupled with MaxPooling2D windows ($2\times2$, padding same).Latent Bottleneck Space: Compresses the pixel grid into a highly dense (None, 7, 7, 32) information matrix.

🔍 Key Insights and Practical ChallengesInformation Bottleneck Function: Because random environmental distortion features lack unified spatial dependency, they are filtered out at the tight $(7,7,32)$ latent bottleneck layer.Loss Gradients: binary_crossentropy consistently outperformed standard Mean Squared Error (MSE) in tracking clear digit definitions on normalized grayscale maps.Reconstruction Limits: A minimal amount of structural softening is visible around complex loops. This trade-off is an expected limitation of an information bottleneck, where the global geometric envelope takes priority over pixel-perfect sharpness.