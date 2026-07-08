# Adversarial-Image-Sandbox

This project is an adversarial machine learning pipeline built from scratch in TensorFlow/Keras. It explores the mathematical vulnerabilities of Convolutional Neural Networks (CNNs) and demonstrates the steep trade-offs required to secure them.

The sandbox trains a baseline VGG architecture on the CIFAR-10 dataset, weaponsizes FGSM and PGD attacks to completely break the model, and then implements a custome Adversarial Training Loop to vaccinate the network against future attacks.


Engineering the Sandbox
1. The Vulnerability
  Standard neural networks are vulnerable to  $L_\infty$ norm attacks. By utilizing Projected Gradient Descent (PGD), I applied invisible, mathematically calculate noise to the images.
  The Result: The baseline model, which achieved ~84.7% accuracy, collapsed to 0.00% accuracy under a PGD attack with an epsilon of just 8/255.

3. The Defense
  To remedy this, I abandoned the standard .fit() method and built a customm adversarial training loop.
  - During training, the pipeline splits every batch 50/50. Half of the images remain clean, while the other half are dynamically attacked using PGD before the optimization step.
  - Fixing Batch Normalization: Initially, I trained on 100% adversarial images, which caused the Batch Normalization statistics to severely drift toward the adversarial distribution, tanking clean accuracy. Implementing the 50/50 split stabilized the running mean/variance, allowing the model to understand both clean and hostile data distributions.
  - 
3. The Tradeoff
Securing the model against PGD attacks required a steep tax on clean accuracy. My evaluation matric proves this trade off: the robust model sacrifices clean performance to act as a sturdy shield when the environment turns hostile.
Evaluation Matrix
(Note: Update these numbers with final run)

Pipeline Usage
To reproduce the results, run the pipeline scripts in the following order:
1. Train the undefended baseline model on CIFAR-10
python3 src/train.py

2. Train the robust model using the custom 50/50 adversarial loop
python3 src/defense.py

3. Mount FGSM and PGD attacks against both models and generate metrics
python3 src/evaluate.py

Tech Stack
- Python 3.11
- TensorFlow / Keras 3 (Custom Training Loops, tf.GradientTape)
- NumPy
