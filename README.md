# Generative Adversarial Networks (GANs) Implementation

## Overview

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed to generate synthetic data that mimics real data. Introduced by Ian Goodfellow et al. in 2014, GANs consist of two neural networks:

1. **Generator (G):** The generator creates synthetic data samples.
2. **Discriminator (D):** The discriminator evaluates whether the samples are real or generated.

The two networks compete in a zero-sum game:
- The generator tries to fool the discriminator into classifying its outputs as real.
- The discriminator learns to distinguish real samples from fake ones.

This adversarial process helps the generator improve over time, creating highly realistic data.

---

## Repository Structure
This repository contains implementations of three types of GAN architectures:

1. **Vanilla GAN**
2. **Deep Convolutional GAN (DCGAN)**
3. **Conditional GAN (CGAN)**
4. **Wasserstain GAN (WGAN)**

Each implementation is provided in a separate Jupyter Notebook:
- `VanillaGAN.ipynb`: Basic implementation of a GAN.
- `DCGAN.ipynb`: Implementation of a convolutional GAN for generating high-quality images.
- `Conditional-GAN.ipynb`: GAN conditioned on labels for controlled data generation.
- `W-GAN.ipynb`: Implementation of Wasserstein GAN. 

---

## What are GANs?
GANs are built upon the idea of training two networks in opposition. The generator's objective is to create data that resembles the training data, while the discriminator's goal is to identify fake samples. The adversarial training dynamic pushes both networks to improve until the generator produces data indistinguishable from real samples.

**Mathematical Objective:**  
The GAN framework is optimized using the following minimax game:

\[
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
\]

Where:  
- \( x \): Real data sample  
- \( z \): Random noise sampled from a prior distribution  
- \( G(z) \): Generated data sample  
- \( D(x) \): Probability that \( x \) is real  

---

## Types of GAN Architectures

### 1. **Vanilla GAN**
The basic GAN architecture consists of fully connected layers in both the generator and discriminator. 

**Key Features:**
- Suitable for low-dimensional data.
- Can suffer from issues like mode collapse, where the generator produces limited diversity in outputs.

[View Implementation](./VanillaGAN.ipynb)

---

### 2. **Deep Convolutional GAN (DCGAN)**
DCGANs introduce convolutional layers into GANs, enabling them to handle high-dimensional data such as images.

**Key Features:**
- Use of convolutional layers in both the generator and discriminator.
- Batch normalization to stabilize training.
- ReLU activation in the generator and Leaky ReLU in the discriminator.
- Produces higher quality images compared to Vanilla GAN.

[View Implementation](./DCGAN.ipynb)

---

### 3. **Conditional GAN (CGAN)**
CGANs extend the GAN framework by conditioning the generation process on auxiliary information (e.g., class labels).

**Key Features:**
- Generator and discriminator receive both noise and class labels as inputs.
- Enables controlled generation of samples corresponding to specific labels.
- Widely used for tasks like image-to-image translation and text-to-image synthesis.

[View Implementation](./Conditional-GAN.ipynb)

---

### 4. **Wasserstein GAN (WGAN)**  
Wasserstein GANs improve upon the original GAN framework by addressing issues like mode collapse and training instability using the Wasserstein distance as a new loss function.

**Key Features:**  
- Replaces the standard GAN loss with the Wasserstein loss, which provides a more meaningful measure of the distance between the real and generated data distributions.  
- Does not require the discriminator to output probabilities; instead, it outputs a critic score.  
- Uses weight clipping or gradient penalty (in WGAN-GP) to enforce the Lipschitz constraint.  
- Improves training stability and makes convergence easier to interpret.

[View Implementation](./W-GAN.ipynb)


---

## Requirements
To run the notebooks, you will need the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

Install the dependencies using:
```bash
pip install torch torchvision numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/AnshChoudhary/GAN-Implementation.git
```

2. Navigate to the repository:

```bash
cd GAN-Implementation
```

3. Open the desired notebook in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook VanillaGAN.ipynb
```

## Applications of GANs
- **Image generation:** Generating realistic faces, artworks, and other synthetic images.
- **Image-to-image translation:** Tasks like style transfer, super-resolution, and domain adaptation.
- **Data augmentation:** Enhancing datasets for machine learning models.
- **Video generation and prediction:** Synthesizing realistic video sequences or predicting future frames.
- **Anomaly detection:** Detecting fraud or other outliers by modeling normal data distributions.

---

## References
1. Goodfellow, I., et al. (2014). Generative Adversarial Networks. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. Radford, A., et al. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
3. Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
4. Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. [arXiv:1701.07875](https://arxiv.org/pdf/1701.07875)

---

## Acknowledgments
This repository was created as a comprehensive guide to GAN architectures and their implementations. Contributions and suggestions are welcome!
