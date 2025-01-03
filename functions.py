import numpy as np

class Activation:
    def __init__(self, activation):
        self.activation = activation

    def forward(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "softmax":
            exps = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exps / np.sum(exps, axis=0, keepdims=True)
        else:
            raise ValueError("Unsupported activation")

    def derivative(self, x, grad_output=None):
        if self.activation == "tanh":
            return 1.0 - np.tanh(x) ** 2
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)
        elif self.activation == "softmax":
            # Softmax gradient is computed within the loss function
            return x
        else:
            raise ValueError("Unsupported activation")

def cross_entropy_loss(predictions, targets):
    num_samples = predictions.shape[1]
    predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    correct_logprobs = -np.log(predictions[targets, np.arange(num_samples)])
    loss = np.sum(correct_logprobs) / num_samples
    return loss

def cross_entropy_derivative(predictions, targets):
    num_samples = predictions.shape[1]
    grad = predictions.copy()
    grad[targets, np.arange(num_samples)] -= 1
    grad /= num_samples
    return grad
