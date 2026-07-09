# 🛡️ Adversarial Image Sandbox

This project is an adversarial machine learning pipeline built from scratch in TensorFlow/Keras. It explores the mathematical vulnerabilities of Convolutional Neural Networks (CNNs) and demonstrates the steep trade-offs required to secure them.

The sandbox trains a baseline VGG architecture on the CIFAR-10 dataset, weaponizes **FGSM** and **PGD** attacks to completely break the model, and then implements a custom **Adversarial Training Loop** to vaccinate the network against future attacks.

---

## 🧠 Engineering the Sandbox

### 1. The Vulnerability (Epsilon Balls)
Standard neural networks are highly vulnerable to $L_\infty$ norm attacks. By utilizing **Projected Gradient Descent (PGD)**, I applied invisible, mathematically calculated noise to the images.
* **The Result:** The baseline model, which achieved ~84.7% clean accuracy, collapsed to **0.00% accuracy** under a PGD attack with an epsilon budget of just $8/255$.

### 2. The Defense (Custom Training Loop)
To remedy this, I abandoned the standard `.fit()` method and built a custom adversarial training loop.
* **The 50/50 Split:** During training, the pipeline splits every batch 50/50. Half of the images remain clean, while the other half are dynamically attacked using PGD before the optimization step.
* **Fixing Batch Normalization Drift:** Initially, I trained on 100% adversarial images. This caused the Batch Normalization statistics to severely drift toward the adversarial distribution, tanking clean accuracy. Implementing the 50/50 split stabilized the running mean/variance, allowing the model to correctly understand both clean and hostile data distributions.

### 3. The "No Free Lunch" Trade-off
Securing the model against PGD attacks required a steep tax on clean accuracy. My evaluation matrix mathematically proves this trade-off: the robust model sacrifices clean performance to act as a sturdy shield when the environment turns hostile.

---

## 📊 Evaluation Matrix

| Model Type | Attack Type | Epsilon=0 | Epsilon=2/255 | Epsilon=8/255 | Epsilon=16/255 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline VGG** | FGSM | 84.70% | 30.60% | 5.30% | 5.30% |
| **Baseline VGG** | PGD | 84.70% | 16.10% | **0.00%** | 0.00% |
| **Robust VGG** | FGSM | 70.40% | 60.70% | 35.60% | 14.20% |
| **Robust VGG** | PGD | 70.40% | 60.50% | **30.50%** | 10.20% |

---

## ⚙️ Pipeline Usage

To reproduce these results, run the pipeline scripts in the following order:

**1. Train the undefended baseline model on CIFAR-10**
```bash
python3 src/train.py
```
2. Train the robust model using the custom 50/50 adversarial loop
```bash
python3 src/defense.py
```
3. Mount FGSM and PGD attacks against both models and generate metrics
```bash
python3 src/evaluate.py
```
🛠️ Tech Stack
Python 3.11

TensorFlow / Keras 3 (Custom Training Loops, tf.GradientTape)

NumPy
