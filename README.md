# Robustness Analysis of Neural Networks (MNIST, MLP)

## 📌 Overview

This project explores the robustness of neural networks under different types of input perturbations.

We implement a Simple MLP on MNIST and analyze robustness from multiple perspectives:

* Sensitivity analysis (gradients & Jacobians)
* Analytical robustness (Lipschitz bounds)
* Adversarial attacks (FGSM, PGD, C&W, DeepFool)
* Sampling-based robustness (random noise)

---

## 🧠 Model

* Dataset: MNIST (28×28 grayscale images)
* Model: 3-layer MLP
* Accuracy: ~97%

---

## 🔬 Methods

### 1. Sensitivity Analysis

* Vanilla gradients
* SmoothGrad
* Integrated Gradients

### 2. Analytical Robustness

* Local linearization
* Jacobian spectral norm
* Robustness radius estimation

### 3. Adversarial Attacks

* FGSM
* PGD (L∞ and L2)
* DeepFool
* Carlini & Wagner (C&W)

### 4. Sampling-Based Methods

* Gaussian noise
* Uniform noise
* Salt & Pepper noise
* Monte Carlo simulation

---

## 🚀 How to Run

```bash
pip install torch torchvision numpy matplotlib
python main.py --run-train
python main.py --run-sensitivity
python main.py --run-robustness
python main.py --run-adversarial
python main.py --run-sampling
```

---

## 📊 Key Insights

* Strong performance under random noise
* High vulnerability to adversarial attacks (especially L∞)
* Lipschitz-based bounds are useful but conservative

---

## 📄 Report

See full report:

```
report.pdf
```

---

## 👥 Authors

EE5311 Group 18
National University of Singapore
