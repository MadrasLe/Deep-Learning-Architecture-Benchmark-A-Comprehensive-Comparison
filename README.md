# Comparative Study of Deep Learning Architectures on MNIST

## Abstract

This project presents a comprehensive benchmark of various Deep Learning architectures implemented in PyTorch, applied to the MNIST handwritten digit classification task. The study covers a wide spectrum of models, including classical **Convolutional Neural Networks (CNNs)**, **Residual Networks (ResNets)**, **Vision Transformers (ViTs)**, **Spiking Neural Networks (SNNs)**, and Generative models like **GANs** and **VAEs**.

The goal is to evaluate and compare these distinct approaches not just on accuracy, but also on training efficiency and model complexity.

## Methodology

*   **Dataset:** Standard MNIST (60,000 training images, 10,000 test images).
*   **Task:** Multi-class classification (10 digits).
*   **Metric:** Top-1 Validation/Test Accuracy.
*   **Environment:**
    *   Framework: PyTorch, Torchvision, Norse (for SNNs).
    *   Hardware: Experiments were primarily conducted on NVIDIA CUDA-capable GPUs, with the exception of SNNs which were trained on CPU.

## Experimental Results

The following table summarizes the key performance metrics extracted from the training logs.

| Model | Architecture Detail | Best Accuracy | Params (approx.) | Time / Epoch | Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet** | MicroResNet (3 stages) | **99.31%** | 174,970 | ~21s | CUDA |
| **CNN** | Standard 2-Layer ConvNet | 98.82% | ~207,000 | ~14s | CUDA |
| **SNN** | Spiking Neural Network | 98.56% | N/A | ~1750s | CPU |
| **DBN** | Deep Belief Network (RBM Stack) | 98.44% | ~670,000 | ~8s (fine-tune) | CUDA |
| **MAPSIAM** | SimSiam Self-Supervised + GBM | 98.24% | N/A | ~47s (pre-train) | CUDA |
| **MLP** | 3-Layer Perceptron (784-512-256-10) | 97.81% | 535,818 | ~15s | CUDA |
| **ViT** | Vision Transformer (Patch Size 4) | 96.97% | 540,170 | ~24.5s | CUDA |
| **GAN** | Discriminator as Classifier | 95.54% | N/A | ~12s | CUDA |
| **U-Net** | Encoder-Decoder (Denoising Pre-train) | 93.93% | N/A | ~13s | CUDA |
| **B-VAE** | Beta-VAE (Latent Classification) | 88.66% | N/A | ~6s | CUDA |

> **Note:** *Time / Epoch* refers to the average training time per epoch as recorded in the logs. *Params* refers to the total number of trainable parameters.

## Analysis & Discussion

### 1. Convolutional Dominance
The **ResNet** and **CNN** models achieved the highest accuracies (>99%), reaffirming that inductive biases local to images (convolutions) remain the most efficient approach for small-scale image datasets like MNIST. The **MicroResNet** was particularly impressive, achieving the best result with the fewest parameters (~175k) among the dense models, highlighting the efficiency of residual connections.

### 2. Transformers on Small Data
The **Vision Transformer (ViT)** reached ~97% accuracy. While respectable, it fell short of the CNN-based models. This illustrates a known characteristic of Transformers: they lack the inherent inductive bias of convolutions (translation invariance) and typically require significantly larger datasets (e.g., ImageNet, JFT) or strong data augmentation to fully outperform CNNs. On MNIST, the overhead in parameter count (~540k) did not yield superior performance compared to the lighter ResNet.

### 3. Spiking Neural Networks (SNN)
The **SNN** implementation demonstrated high accuracy (98.56%), proving that biologically-inspired spiking models can compete with traditional ANNs on static tasks. However, the computational cost on standard hardware (CPU) was massive (~30 minutes/epoch vs. 15 seconds/epoch for ANNs). This highlights the need for neuromorphic hardware or optimized GPU kernels for practical SNN deployment.

### 4. Self-Supervised & Generative Approaches
*   **MAPSIAM (SimSiam)** achieved >98% accuracy using a simple gradient boosting classifier on top of learned embeddings. This confirms that self-supervised learning can extract robust features without label information during the feature learning phase.
*   **Generative Models (GAN, VAE, DBN)** generally lagged behind purely discriminative models in classification accuracy. This is expected, as their objective functions optimize for data generation/reconstruction (p(x)) rather than class discrimination (p(y|x)). However, the DBN's strong performance (98.44%) shows the value of layer-wise pre-training.

## Conclusion

For the MNIST task, **MicroResNet** strikes the best balance between accuracy, parameter efficiency, and training speed. While newer architectures like ViT and SNNs are theoretically powerful, they require specific conditions (more data or specialized hardware) to shine.

## Dependencies

To replicate these experiments, the following libraries are required:
*   `python >= 3.12`
*   `torch` & `torchvision`
*   `norse` (Spiking Neural Networks)
*   `nir` / `nirtorch`
*   `lightgbm` (for MAPSIAM evaluation)
*   `tqdm` (for progress tracking)

## License

This project is licensed under the MIT License.
