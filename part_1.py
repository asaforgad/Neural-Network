import numpy as np
import matplotlib.pyplot as plt
import scipy

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Compute the softmax regression loss
def softmax_regression(X, C, W, b, probs=None):
    m = X.shape[0]
    if probs is None:
        logits = X @ W + b
        probs = softmax(logits)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    # Compute cross-entropy loss
    loss = -np.sum(C * np.log(probs)) / m
    return loss

# Compute the gradient of the softmax regression loss
def softmax_gradient(X, C, W, b, probs=None):
    m = X.shape[0]
    if probs is None:
        logits = X @ W + b
        probs = softmax(logits)
    grad_W = (X.T @ (probs - C)) / m
    grad_b = np.sum(probs - C, axis=0, keepdims=True) / m
    return grad_W, grad_b

# Gradient Test
def gradient_test():
    samples, features, classes = 20, 10, 5  # Samples, features, classes
    # Random data for testing
    np.random.seed(42)
    X = np.random.randn(samples, features)  # input data
    W = np.random.randn(features, classes)
    b = np.random.randn(1, classes)
    y = np.random.randint(0, classes, size=samples)
    C = np.zeros((samples, classes))  # One-hot encoded labels
    C[np.arange(samples), y] = 1

    # Random perturbations
    dW = np.random.randn(*W.shape)
    db = np.random.randn(*b.shape)
    epsilon = 0.1

    # Compute initial loss and gradients
    F0 = softmax_regression(X, C, W, b)
    grad_W, grad_b = softmax_gradient(X, C, W, b)

    # Initialize error storage
    zero_order_error_W, first_order_error_W = [], []
    zero_order_error_b, first_order_error_b = [], []

    # Perform gradient test for W
    for k in range(1, 9):
        epsk = epsilon * (0.5**k)
        Fk_W = softmax_regression(X, C, W + epsk * dW, b)
        F1_W = F0 + epsk * np.sum(grad_W * dW)
        zero_order_error_W.append(abs(Fk_W - F0))
        first_order_error_W.append(abs(Fk_W - F1_W))

    # Perform gradient test for b
    for k in range(1, 9):
        epsk = epsilon * (0.5**k)
        Fk_b = softmax_regression(X, C, W, b + epsk * db)
        F1_b = F0 + epsk * np.sum(grad_b * db)
        zero_order_error_b.append(abs(Fk_b - F0))
        first_order_error_b.append(abs(Fk_b - F1_b))

    # Plot errors for W
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(range(1, 9), zero_order_error_W, label="Zero order (W)")
    plt.semilogy(range(1, 9), first_order_error_W, label="First order (W)")
    plt.title("Gradient Test for W")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    # Plot errors for b
    plt.subplot(1, 2, 2)
    plt.semilogy(range(1, 9), zero_order_error_b, label="Zero order (b)")
    plt.semilogy(range(1, 9), first_order_error_b, label="First order (b)")
    plt.title("Gradient Test for b")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Least Squares Loss Function
def least_squares_loss(X, y, weights, bias):
    predictions = X @ weights + bias
    losses = 0.5 * ((predictions - y) ** 2)
    return np.mean(losses)

# Least Squares Gradient Function
def least_squares_gradient(X, y, weights, bias):
    samples_number = len(y)
    predictions = X @ weights + bias
    residuals = predictions - y
    grad_weights = X.T @ residuals / samples_number
    grad_bias = np.sum(residuals) / samples_number
    return grad_weights, grad_bias

def stochastic_gradient_descent_least_squares(X, y, learning_rate=0.5, epochs=500, batch_size=200, weights=None, bias=None, gradient_function=least_squares_gradient , loss_function=least_squares_loss):
    samples_number = X.shape[0]
    losses = []

    if weights is None:
        weights = np.random.randn(X.shape[1],1)
    if bias is None:
        bias = np.random.randn(1)

    for epoch in range(epochs):
        # Shuffle the data
        random_index = np.random.permutation(samples_number)
        X = X[random_index]
        y = y[random_index]

        # Mini-batch gradient descent
        for batch_start in range(0, samples_number, batch_size):
            X_batch = X[batch_start:batch_start + batch_size]
            y_batch = y[batch_start:batch_start + batch_size]

            # Compute gradients
            gradient_weights, gradient_bias = gradient_function(X_batch, y_batch, weights, bias)

            # Update weights and bias
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias

        # Append loss after epoch
        loss = loss_function(X, y, weights, bias)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


    return weights, bias, losses

# test SGD optimizer
def test_sgd_least_squares():
  np.random.seed(42)
  X = np.random.rand(100, 1)
  true_weights = np.array([[2.0]])
  true_bias = 3.0
  y = X @ true_weights + true_bias + 0.1 * np.random.randn(100, 1)

  weights, bias, losses = stochastic_gradient_descent_least_squares(X, y, learning_rate=0.05, epochs=100, batch_size=8)
  print("Estimated Weights:", weights.flatten())
  print("Estimated Bias:", bias)

  plt.plot(losses)
  plt.title("Loss vs. Epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid()
  plt.show()

  plt.scatter(X, y, label="Data Points", alpha=0.7)
  plt.plot(X, X @ weights + bias, color='red', label="Fitted Line", linewidth=2)
  plt.title("Linear Fit Using SGD")
  plt.xlabel("X")
  plt.ylabel("y")
  plt.legend()
  plt.grid()
  plt.show()

  mse = least_squares_loss(X, y, weights, bias)
  print("Mean Squared Error:", mse)

# SGD Implementation
def sgd(X, C, learning_rate=0.01, batch_size=200, epochs=1000, x_val=None, y_val=None,
        weights=None, biases=None, plot=False, print_progress=False):
    n_features = X.shape[1]
    n_classes = C.shape[1]

    if weights is None:
        weights = np.random.randn(n_features, n_classes)
    if biases is None:
        biases = np.random.randn(1, n_classes)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_train = X[indices]
        C_train = C[indices]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            C_batch = C_train[i:i + batch_size]

            logits = X_batch @ weights + biases
            probs = softmax(logits)
            grad_W, grad_b = softmax_gradient(X_batch, C_batch, weights, biases, probs)

            weights -= learning_rate * grad_W
            biases -= learning_rate * grad_b

        # Compute training accuracy
        logits_train = X @ weights + biases
        pred_train = np.argmax(logits_train, axis=1)
        true_train = np.argmax(C, axis=1)
        train_accuracy = np.mean(pred_train == true_train)
        train_accuracies.append(train_accuracy)

        # Compute test accuracy (if validation data provided)
        if x_val is not None and y_val is not None:
            logits_val = x_val @ weights + biases
            pred_val = np.argmax(logits_val, axis=1)
            true_val = np.argmax(y_val, axis=1)
            val_accuracy = np.mean(pred_val == true_val)
            val_accuracies.append(val_accuracy)

        if print_progress and (epoch + 1) % 10 == 0 or epoch == 0:
            if x_val is not None and y_val is not None:
                print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_accuracy:.4f}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
        if val_accuracies:
            plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.title('SGD Training Progress')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    return weights, biases, train_accuracies, val_accuracies

# Helper function to find the best learning rate and batch size for SGD
def find_best_learning_rates_and_batch_sizes(X_train, C_train, x_val=None, y_val=None,iterations=20, plot_graph=False, print_each_iteration=False):
    best_accuracy = 0
    best_lr = None
    best_batch_size = None
    learning_rates = [0.1, 0.01, 0.001, 0.5]
    batch_sizes = [32, 64, 91, 128, 181, 200, 256]

    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            weights, biases, train_accuracies, val_accuracies = sgd(
                X_train, C_train,
                learning_rate=lr,
                batch_size=bs,
                epochs=iterations,
                x_val=x_val,
                y_val=y_val,
                plot=False,
                print_progress=False
            )

            if val_accuracies:
                last_val_accuracy = val_accuracies[-1]  # הדיוק האחרון
                if last_val_accuracy > best_accuracy:
                    best_accuracy = last_val_accuracy
                    best_lr = lr
                    best_batch_size = bs

            result = {
                "learning_rate": lr,
                "batch_size": bs,
                "val_accuracy": val_accuracies[-1] if val_accuracies else None,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
            }
            results.append(result)

            if print_each_iteration:
              print(f"Learning rate: {lr}, Batch size: {bs}, Validation Accuracy: {val_accuracies[-1]:.4f}" if val_accuracies else f"Learning rate: {lr}, Batch size: {bs}")

    print(f"\nBest Learning Rate: {best_lr}, Best Batch Size: {best_batch_size}, Best Accuracy: {best_accuracy:.4f}")

    if plot_graph:
        lr_labels = [f"LR={result['learning_rate']}, BS={result['batch_size']}" for result in results]
        accuracies = [result["val_accuracy"] for result in results if result["val_accuracy"] is not None]

        plt.figure(figsize=(12, 6))
        plt.bar(lr_labels, accuracies, color='skyblue')
        plt.xlabel("Learning Rate and Batch Size")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy for Different Learning Rates and Batch Sizes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        for result in results:
            if result["learning_rate"] == best_lr and result["batch_size"] == best_batch_size:
                plt.figure(figsize=(10, 5))
                plt.plot(result["train_accuracies"], label="Training Accuracy")
                plt.plot(result["val_accuracies"], label="Validation Accuracy")
                plt.title(f"Accuracy Over Epochs (Best LR={best_lr}, BS={best_batch_size})")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True)
                plt.show()

    return best_lr, best_batch_size

