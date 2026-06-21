import os
import tensorflow as tf
import keras
from keras.datasets import cifar10
from model import create_vgg_cnn

# Bypasses the macOS SSL certificate check for dataset download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_baseline():
    print("Step 1: Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize inputs strictly to [0, 1] float32
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    print("Step 2: Instantiating VGG model architecture...")
    model = create_vgg_cnn()
    
    # Compile with Adam optimizer and Sparse Categorical Crossentropy
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Step 3: Training model (15 epochs to establish a strong baseline)...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=15,
        batch_size=64
    )
    
    print("\nStep 4: Evaluating final model performance on clean test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"🥇 Clean Baseline Test Accuracy: {test_acc * 100:.2f}%")
    
    # Ensure a directory exists to save our trained weights
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_vgg.keras"
    model.save(model_path)
    print(f"💾 Successfully saved baseline weights to: {model_path}")

if __name__ == "__main__":
    train_baseline()