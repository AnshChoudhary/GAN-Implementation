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
