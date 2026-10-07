# Adversarial Image Sandbox

![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-3-orange)
![License](https://img.shields.io/badge/license-MIT-green)
<!-- After you add .github/workflows/ci.yml, uncomment and fix the path:
![CI](https://github.com/jaydenxaf/Adversarial-Image-Sandbox/actions/workflows/ci.yml/badge.svg)
-->

An adversarial machine learning pipeline built from scratch in TensorFlow/Keras. It trains a VGG-style CNN on CIFAR-10, breaks it with **FGSM** and **PGD** attacks, and then hardens it with a custom **adversarial training loop**, measuring exactly what that robustness costs in clean accuracy.

![Accuracy vs. epsilon for baseline and robust models](results/accuracy_vs_epsilon.png)
<!-- TODO: generate this figure (accuracy vs. epsilon, 4 lines: baseline/robust x FGSM/PGD) and save it to results/ -->

![Clean vs. adversarial examples](results/adversarial_examples.png)
<!-- TODO: generate a row of clean image | perturbation (amplified) | adversarial image, with predicted labels -->

## Key Results

- **The undefended model is completely broken.** The baseline reaches **84.7%** clean accuracy but drops to **0.00%** under PGD at ε = 8/255, a perturbation invisible to a human.
- **Adversarial training works, at a cost.** The robust model holds **30.5%** accuracy under the same PGD attack, but clean accuracy falls from 84.7% to **70.4%** (about 14 points).
- **There is no free lunch.** Robustness is bought with clean accuracy, and the evaluation matrix below quantifies the trade.

## Evaluation Matrix

Test accuracy by attack strength (ε is the L∞ perturbation budget on pixel values scaled to [0, 1]; 8/255 ≈ 0.031).

| Model | Attack | ε = 0 | ε = 2/255 | ε = 8/255 | ε = 16/255 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline VGG | FGSM | 84.70% | 30.60% | 5.30% | 5.30% |
| Baseline VGG | PGD | 84.70% | 16.10% | **0.00%** | 0.00% |
| Robust VGG | FGSM | 70.40% | 60.70% | 35.60% | 14.20% |
| Robust VGG | PGD | 70.40% | 60.50% | **30.50%** | 10.20% |

**Evaluation setup:** [TODO: number of test images used (the one-decimal values suggest ~1,000), random seed(s), and whether results are a single run or an average].

## Background

Standard CNNs are vulnerable to small, carefully computed perturbations bounded in the L∞ norm: every pixel may change by at most ε, so the "epsilon ball" around an image looks unchanged to a person but can flip the model's prediction.

**FGSM** (Fast Gradient Sign Method) takes a single step in the direction that increases the loss:

$$x_{adv} = \text{clip}\big(x + \varepsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))\big)$$

**PGD** (Projected Gradient Descent) is the iterative version of FGSM. It takes several smaller steps and, after each one, projects the result back into the ε-ball around the original image. It is a much stronger attack than FGSM.

## Method

### Attacks (`src/attacks.py`)

| Parameter | FGSM | PGD |
| :--- | :--- | :--- |
| Norm | L∞ | L∞ |
| Steps | 1 | [TODO] |
| Step size (α) | n/a | [TODO] |
| Random start | n/a | [TODO: yes/no] |

### Defense: custom adversarial training loop (`src/defense.py`)

Instead of `model.fit()`, the defense uses a custom loop built on `tf.GradientTape`. For every batch:

1. Split the batch 50/50.
2. Leave one half clean.
3. Attack the other half with PGD (ε = [TODO: training ε]) against the current model.
4. Recombine the halves and take one optimization step on the mixed batch.

**Why 50/50 and not 100% adversarial?** My first version trained only on adversarial images. That shifted the Batch Normalization running mean/variance toward the adversarial distribution and hurt clean accuracy. Mixing clean and adversarial images in every batch keeps the BN statistics representative of both distributions.

| Training data per batch | Clean accuracy | PGD accuracy (ε = 8/255) |
| :--- | :--- | :--- |
| 100% adversarial | [TODO] | [TODO] |
| 50% clean / 50% adversarial | 70.40% | 30.50% |

<!-- TODO: fill in the 100% adversarial row from a real rerun before keeping the BN claim. -->

## Quick Start

```bash
git clone https://github.com/jaydenxaf/Adversarial-Image-Sandbox.git
cd Adversarial-Image-Sandbox
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically by Keras on first run. Trained models are saved as `.keras` files, which are gitignored, so run the scripts in order:

```bash
# 1. Train the undefended baseline model on CIFAR-10
python3 src/train.py

# 2. Train the robust model with the custom 50/50 adversarial loop
python3 src/defense.py

# 3. Run FGSM and PGD against both models and generate metrics
python3 src/evaluate.py
```

For a guided tour, open [`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb).

## Repository Structure

```
.
├── src/
│   ├── model.py       # VGG-style architecture
│   ├── train.py       # baseline training
│   ├── attacks.py     # FGSM and PGD implementations
│   ├── defense.py     # custom adversarial training loop
│   └── evaluate.py    # attack evaluation across epsilons
├── notebooks/         # walkthrough and visualizations
├── examples/          # sample image for the demo notebook
├── results/           # figures and metrics
└── README.md
```

## Limitations

- **Threat model:** L∞ perturbations only, with white-box access to the model. L2 attacks, black-box/transfer attacks, and physical-world perturbations are not covered.
- **Attack strength:** Robustness is measured with FGSM and PGD only. A stronger evaluation (for example AutoAttack) would likely report lower robust accuracy, so treat the robust numbers as upper bounds.
- **Single training budget:** The robust model is trained against one ε, and accuracy falls off beyond it (10.2% under PGD at 16/255).
- **Scope:** CIFAR-10 (32×32 images) with one architecture family. Results may not transfer to larger datasets or models.

## Roadmap

- [ ] Verify the Batch Normalization fix across multiple random seeds
- [ ] Add a stronger evaluation (AutoAttack)
- [ ] Add L2-norm attacks
- [ ] Test attack transferability between the baseline and robust models
- [ ] Add unit tests (attack stays inside the ε-ball and the valid pixel range)

## References

- Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples.* (FGSM)
- Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* (PGD adversarial training)
- Simonyan, Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* (VGG)

<!-- TODO: verify titles, authors, and years, then add links. -->

## Tech Stack

Python 3.11 · TensorFlow / Keras 3 (custom training loops, `tf.GradientTape`) · NumPy · Matplotlib

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Author

Built by [@jaydenxaf](https://github.com/jaydenxaf).
