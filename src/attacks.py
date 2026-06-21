import tensorflow as tf

def fast_gradient_sign_method(model, images, labels, epsilon):
    """
    Generates an adversarial perturbation using a single-step FGSM attack.
    """
    # Force TensorFlow to track operations on the input images
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        
    # Calculate the gradient of the loss with respect to the input pixels
    gradient = tape.gradient(loss, images)
    
    # Take the sign of the gradient
    signed_grad = tf.sign(gradient)
    
    # Construct the adversarial image
    adversarial_images = images + epsilon * signed_grad
    
    # Clip the pixel values to keep them strictly in the valid [0, 1] range
    return tf.clip_by_value(adversarial_images, 0.0, 1.0)


def projected_gradient_descent(model, images, labels, epsilon, alpha=2/255, num_iter=10):
    """
    Generates an adversarial perturbation using iterative PGD with clipping.
    """
    # Start with a copy of the original clean images
    adv_images = tf.identity(images)
    
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            
        # Get the gradient relative to the current adversarial image state
        gradient = tape.gradient(loss, adv_images)
        signed_grad = tf.sign(gradient)
        
        # Take a tiny step in the malicious direction
        adv_images = adv_images + alpha * signed_grad
        
        # --- THE PROJECTION STEP ---
        # 1. Calculate how far we drifted from the original clean image
        perturbation = adv_images - images
        # 2. Clip that drift so it never exceeds our maximum epsilon budget
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        # 3. Add the safe perturbation back to the original image
        adv_images = images + perturbation
        
        # Final safety constraint: ensure pixel array elements stay valid colors [0, 1]
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
        
    return adv_images

if __name__ == "__main__":
    print("Attacks module compiled successfully. Ready for import.")