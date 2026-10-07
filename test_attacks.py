"""Invariant tests for the FGSM and PGD attacks in src/attacks.py.

An L-infinity attack with budget epsilon must satisfy, for every pixel:
    |x_adv - x| <= epsilon    (stays inside the epsilon ball)
    0 <= x_adv <= 1           (stays a valid image)
These tests use a tiny randomly initialised model so they run in seconds on CPU.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers

from src.attacks import fast_gradient_sign_method, projected_gradient_descent

# Float32 rounding can push |x_adv - x| a hair past epsilon
TOL = 1e-6
EPSILONS = [2 / 255, 8 / 255, 16 / 255]


@pytest.fixture(autouse=True)
def _seed():
    """Make random starts and weight init reproducible."""
    keras.utils.set_random_seed(0)
    tf.random.set_seed(0)


@pytest.fixture(scope="module")
def model():
    keras.utils.set_random_seed(0)
    return keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(4, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ]
    )


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    return tf.constant(rng.random((8, 32, 32, 3), dtype=np.float32))


@pytest.fixture(scope="module")
def labels():
    rng = np.random.default_rng(1)
    return tf.constant(rng.integers(0, 10, size=8), dtype=tf.int32)


def run_attack(name, model, x, y, eps, **pgd_kwargs):
    if name == "fgsm":
        return fast_gradient_sign_method(model, x, y, eps)
    kwargs = {"alpha": eps / 4, "num_iter": 10}
    kwargs.update(pgd_kwargs)
    return projected_gradient_descent(model, x, y, eps, **kwargs)


def linf(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_output_shape_and_dtype_match_input(attack, model, images, labels):
    adv = run_attack(attack, model, images, labels, 8 / 255)
    assert adv.shape == images.shape
    assert adv.dtype == images.dtype


@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_perturbation_stays_inside_epsilon_ball(attack, eps, model, images, labels):
    adv = run_attack(attack, model, images, labels, eps)
    assert linf(adv, images) <= eps + TOL


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_attack_actually_perturbs_the_image(attack, model, images, labels):
    adv = run_attack(attack, model, images, labels, 8 / 255)
    assert linf(adv, images) > 0


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_output_stays_in_valid_pixel_range(attack, model, images, labels):
    adv = np.asarray(run_attack(attack, model, images, labels, 16 / 255))
    assert adv.min() >= 0.0
    assert adv.max() <= 1.0


@pytest.mark.parametrize("fill", [0.0, 1.0])
@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_saturated_pixels_are_clipped(attack, fill, model, labels):
    """All-black / all-white images must not be pushed outside [0, 1]."""
    x = tf.fill((8, 32, 32, 3), tf.constant(fill, dtype=tf.float32))
    adv = np.asarray(run_attack(attack, model, x, labels, 16 / 255))
    assert adv.min() >= 0.0
    assert adv.max() <= 1.0


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_zero_epsilon_returns_the_original_image(attack, model, images, labels):
    adv = run_attack(attack, model, images, labels, 0.0)
    assert linf(adv, images) <= TOL


def test_pgd_projection_holds_when_step_size_exceeds_epsilon(model, images, labels):
    """Without the projection step, 10 steps of size 2*eps could drift up to 20*eps."""
    eps = 4 / 255
    adv = run_attack("pgd", model, images, labels, eps, alpha=2 * eps, num_iter=10)
    assert linf(adv, images) <= eps + TOL


def test_pgd_increases_loss(model, images, labels):
    """Sanity check: PGD is gradient *ascent*, so loss should go up, not down."""
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    clean_loss = float(loss_fn(labels, model(images)))
    adv = run_attack("pgd", model, images, labels, 8 / 255)
    adv_loss = float(loss_fn(labels, model(adv)))
    assert adv_loss >= clean_loss
