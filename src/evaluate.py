import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from attacks import fast_gradient_sign_method, projected_gradient_descent

def run_evaluation():
    print("🚀 Initializing Pipeline Evaluation...")
    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    
    # Use a subset of 1,000 images for evaluation to make computation fast
    eval_size = 1000
    x_eval = tf.convert_to_tensor(x_test[:eval_size])
    y_eval = tf.convert_to_tensor(y_test[:eval_size])
    
    print("📦 Loading saved models from disk...")
    baseline_model = tf.keras.models.load_model("models/baseline_vgg.keras")
    robust_model = tf.keras.models.load_model("models/robust_vgg.keras")
    
    # Define our testing matrix parameters
    epsilons = [0.0, 2/255, 8/255, 16/255]
    
    print("\n📊 Running adversarial evaluations across epsilon spectrum...")
    print("| Model Type | Attack Type | Epsilon=0 | Epsilon=2/255 | Epsilon=8/255 | Epsilon=16/255 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for model_name, model in [("Baseline VGG", baseline_model), ("Robust VGG", robust_model)]:
        for attack_name in ["FGSM", "PGD"]:
            row_results = [f"{model_name}", f"{attack_name}"]
            
            for eps in epsilons:
                if eps == 0.0:
                    # Epsilon 0 means clean data; no attack needed
                    preds = model.predict(x_eval, verbose=0)
                elif attack_name == "FGSM":
                    x_adv = fast_gradient_sign_method(model, x_eval, y_eval, epsilon=eps)
                    preds = model.predict(x_adv, verbose=0)
                elif attack_name == "PGD":
                    x_adv = projected_gradient_descent(model, x_eval, y_eval, epsilon=eps, alpha=2/255, num_iter=10)
                    preds = model.predict(x_adv, verbose=0)
                
                # Calculate accuracy on the evaluation batch
                acc = np.mean(np.argmax(preds, axis=1) == y_eval.numpy().flatten())
                row_results.append(f"{acc * 100:.2f}%")
            
            print(f"| {' | '.join(row_results)} |")

if __name__ == "__main__":
    run_evaluation()