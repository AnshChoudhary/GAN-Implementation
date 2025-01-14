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

Each implementation is provided in a separate Jupyter Notebook:
- `VanillaGAN.ipynb`: Basic implementation of a GAN.
- `DCGAN.ipynb`: Implementation of a convolutional GAN for generating high-quality images.
- `Conditional-GAN.ipynb`: GAN conditioned on labels for controlled data generation.

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
