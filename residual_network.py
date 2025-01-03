from matplotlib import pyplot as plt
import numpy as np

from functions import cross_entropy_derivative, cross_entropy_loss
from layers import Layer, ResNetLayer


class ResNet:
    def __init__(self, input_size=5, hidden_sizes=[5, 5], output_size=5, activations=None, lr=0.01):
        """
        input_size: Number of neurons in the input layer
        hidden_sizes: List defining the size of each hidden layer
        output_size: Number of neurons in the output layer
                    for example,
        activations: List defining the activation function for each layer (excluding the input layer).
                     If not provided, defaults to 'tanh' for all layers except the output layer, which is 'softmax'.
        lr : learning rate
        
        For example: input_size=5, hidden_sizes=[5, 5], output_size=5 represents:
                        Input layer with 5 neurons,
                        Two hidden layers with 5 neurons each,
                        and Output layer with 5 neurons.
        """
        self.layers = []
        self.num_layers = len(hidden_sizes) + 1
        if activations is None:
            activations = ['tanh'] * (self.num_layers - 1) + ['softmax']
        elif len(activations) != self.num_layers:
            raise ValueError("Length of activations must be equal to number of layers")
        
        self.layers.append(ResNetLayer(input_size, hidden_sizes[0], activation=activations[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(ResNetLayer(input_size, hidden_sizes[i+1], activation=activations[i+1]))
        
        self.layers.append(Layer(input_size, output_size, activation=activations[-1]))
        
        self.lr = lr
        
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X, Y, output):
        grads = {}  # Dictionary to store gradients for all layers
        dA = cross_entropy_derivative(output, Y)  # Loss derivative with respect to the last layer's A
        
        # Backward pass through output layer
        output_layer = self.layers[-1]
        dA_prev, grad_W, grad_b = output_layer.backward(dA)
        grads[f'W{self.num_layers}'] = grad_W
        grads[f'b{self.num_layers}'] = grad_b
        
        # Backward pass through hidden layers
        for idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[idx]
            if isinstance(layer, ResNetLayer):
                dA_prev, grad_W1, grad_b1, grad_W2, grad_b2 = layer.backward(dA_prev)
                grads[f'W1{idx+1}'] = grad_W1
                grads[f'b1{idx+1}'] = grad_b1
                grads[f'W2{idx+1}'] = grad_W2
                grads[f'b2{idx+1}'] = grad_b2
            else:
                dA_prev, grad_W, grad_b = layer.backward(dA_prev)
                grads[f'W{idx+1}'] = grad_W
                grads[f'b{idx+1}'] = grad_b
        
        return grads
    
    def update_params(self):
        # Update parameters for all layers in the network
        for layer in self.layers:
            layer.update_params(self.lr)
            
    def back_propagation(self, X, Y, output):
        self.backward(X, Y, output)
        self.update_params()

    def get_flat_params(self):
        # Return all layers parameters as a single vector
        return np.concatenate([layer.get_params() for layer in self.layers])

    def set_flat_params(self, flat_params):
        # Set all layer parameters from a single vector
        current_idx = 0
        for layer in self.layers:
            layer_params = layer.get_params()
            params_size = layer_params.size
            layer.set_params(flat_params[current_idx:current_idx + params_size])
            current_idx += params_size

    def train(self, X, y, num_epochs, batch_size=32, x_val=None, y_val=None):  # Added learning rate parameter
            if X.shape[1] != y.shape[1]:
                raise ValueError("Mismatch between number of samples in X and y")
            if x_val is not None and y_val is not None:
                if x_val.shape[1] != y_val.shape[1]:
                    raise ValueError("Mismatch between number of samples in x_val and y_val")

            losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            for epoch in range(num_epochs):
                # Shuffle data
                indices = np.random.permutation(X.shape[1])
                X_shuffled = X[:, indices]
                y_shuffled = y[:, indices]

                epoch_loss = 0

                for i in range(0, X_shuffled.shape[1], batch_size):
                    X_batch = X_shuffled[:, i:i + batch_size]
                    y_batch = y_shuffled[:, i:i + batch_size]

                    # Forward pass
                    y_pred = self.forward(X_batch)
                    Y_batch = np.argmax(y_batch, axis=0) # Convert one-hot to integer labels

                    # Compute loss
                    loss = cross_entropy_loss(y_pred, Y_batch)
                    epoch_loss += loss

                    # Backward pass
                    self.backward(X_batch, Y_batch, y_pred)
                    self.update_params()
                    
                average_epoch_loss = epoch_loss / (X_shuffled.shape[1] / batch_size)
                losses.append(average_epoch_loss)

                # Compute training accuracy
                train_pred = self.forward(X)
                train_accuracy = np.mean(np.argmax(train_pred, axis=0) == np.argmax(y, axis=0))
                train_accuracies.append(train_accuracy)

                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_epoch_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")

                # Compute validation loss and accuracy
                if x_val is not None and y_val is not None:
                    val_pred = self.forward(x_val)
                    val_loss = cross_entropy_loss(val_pred, np.argmax(y_val, axis=0))
                    val_losses.append(val_loss)

                    val_accuracy = np.mean(np.argmax(val_pred, axis=0) == np.argmax(y_val, axis=0))
                    val_accuracies.append(val_accuracy)

                    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

            # Plot training and validation metrics side by side
            epochs = range(1, num_epochs + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Loss plot
            axes[0].plot(epochs, losses, label="Training Loss", color="blue")
            if x_val is not None and y_val is not None:
                axes[0].plot(epochs, val_losses, label="Validation Loss", color="orange")
            axes[0].set_title("Loss Vs Epochs")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

            # Accuracy plot
            axes[1].plot(epochs, train_accuracies, label="Training Accuracy", color="blue")
            if x_val is not None and y_val is not None:
                axes[1].plot(epochs, val_accuracies, label="Validation Accuracy", color="orange")
            axes[1].set_title("Accuracy Vs Epochs")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            