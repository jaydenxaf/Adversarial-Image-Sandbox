import os
import tensorflow as tf
import keras
from keras.datasets import cifar10
from model import create_vgg_cnn
from attacks import projected_gradient_descent

# Bypass macOS SSL certificate check
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_robust_model():
    print("Loading CIFAR-10 dataset for robust training...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    print("Instantiating a fresh VGG architecture for defense...")
    robust_model = create_vgg_cnn()
    
    robust_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Custom training loop parameters
    epochs = 10
    batch_size = 128  # Larger batch size to stabilize adversarial gradients
    epsilon = 8/255
    
    print(f"Starting Adversarial Training Loop ({epochs} epochs)...")
    print("Each batch will be perturbed using PGD before optimization step.")
    
    # Convert numpy data into a high-performance TensorFlow Dataset pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Progress tracking metrics
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # 1. Generate adversarial versions of this specific batch on the fly
            x_batch_adv = projected_gradient_descent(
                robust_model, x_batch, y_batch, epsilon=epsilon, alpha=2/255, num_iter=7
            )
            
            # 2. Open a GradientTape to monitor the weights update
            with tf.GradientTape() as tape:
                predictions = robust_model(x_batch_adv, training=True)
                loss_value = robust_model.compute_loss(x_batch_adv, y_batch, predictions)
                
            # 3. Calculate gradients of weights and apply optimizer step
            grads = tape.gradient(loss_value, robust_model.trainable_variables)
            robust_model.optimizer.apply_gradients(zip(grads, robust_model.trainable_variables))
            
            # Update metrics
            epoch_loss.update_state(loss_value)
            epoch_acc.update_state(y_batch, predictions)
            
            if step % 50 == 0:
                print(f"  Step {step}: Loss = {epoch_loss.result():.4f}, Accuracy = {epoch_acc.result()*100:.2f}%")
                
    # Evaluate performance on clean data
    clean_loss, clean_acc = robust_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🛡️ Robust Model Clean Test Accuracy: {clean_acc * 100:.2f}%")
    
    # Save the robust weights distinctly from the baseline
    os.makedirs("models", exist_ok=True)
    robust_path = "models/robust_vgg.keras"
    robust_model.save(robust_path)
    print(f"💾 Successfully saved robust model weights to: {robust_path}")

if __name__ == "__main__":
    train_robust_model()