import numpy as np
from functions import Activation,cross_entropy_loss,cross_entropy_derivative


class Layer:
    def __init__(self, input_size, output_size, activation='tanh'):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size, 1)
        self.activation = activation
        self.activation_func = Activation(activation)
        self.residental = False
        self.dW, self.db, self.input, self.output = None, None, None, None

    def forward(self, A_prev):
        self.input = A_prev
        Z = self.W.T.dot(A_prev) + self.b
        self.output = self.activation_func.forward(Z)
        return self.output

    def backward(self, dA):
        if self.activation_func.activation == "softmax":
            dZ = dA  # Softmax gradient is computed within the loss function
        else:
            dZ = dA * self.activation_func.derivative((self.W.T.dot(self.input) + self.b))

        self.dW = self.input.dot(dZ.T)  
        self.db = np.sum(dZ, axis=1, keepdims=True) 
        dA_prev = self.W.dot(dZ) 
        return dA_prev, self.dW, self.db

    def update_params(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def get_params(self):
        # Return the layer's parameters as a vector
        return np.concatenate([self.W.flatten(), self.b.flatten()])

    def set_params(self, flat_params):
        # Set the layer's parameters from a vector
        W_size = self.W.size
        self.W = flat_params[:W_size].reshape(self.W.shape)
        self.b = flat_params[W_size:].reshape(self.b.shape)
        
        
class ResNetLayer:
    def __init__(self, input_size, output_size, activation='tanh'):
        self.W1 = np.random.randn(input_size, output_size)
        self.b1 = np.random.randn(output_size, 1)
        self.W2 = np.random.randn(output_size, input_size)
        self.b2 = np.random.randn(input_size, 1)
        self.activation_func = Activation(activation)
        self.residental = True
        self.input, self.output = None, None

    def forward(self, A_prev):
        self.input = A_prev
        Z1 = self.W1.T.dot(A_prev) + self.b1
        A1 = self.activation_func.forward(Z1)
        
        Z2 = self.W2.T.dot(A1) + self.b2
        Z2 += self.input
        A2 = self.activation_func.forward(Z2)
        
        self.output = A2
        return self.output


    def backward(self, dA2):
        Z1 = self.W1.T.dot(self.input) + self.b1
        A1 = self.activation_func.forward(Z1)
        Z2 = (self.W2.T.dot(A1) + self.b2) + self.input
        
        dZ2 = dA2 * self.activation_func.derivative(Z2)
        self.dW2 = A1.dot(dZ2.T)
        self.db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.dot(dZ2)
        
        dZ1 = dA1 * self.activation_func.derivative(Z1)
        self.dW1 = self.input.dot(dZ1.T)
        self.db1 = np.sum(dZ1, axis=1, keepdims=True)
        
        dX = self.W1.dot(dZ1) + dZ2
        
        return dX, self.dW1, self.db1, self.dW2, self.db2

    def update_params(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def get_params(self):
        # Return the layer's parameters as a vector
        return np.concatenate([self.W1.flatten(), self.b1.flatten(),
                               self.W2.flatten(), self.b2.flatten()])
        
    def set_params(self, flat_params):
        # Set the layer's parameters from a vector
        W1_size = self.W1.size
        b1_size = self.b1.size
        W2_size = self.W2.size
        b2_size = self.b2.size
        self.W1 = flat_params[:W1_size].reshape(self.W1.shape)
        self.b1 = flat_params[W1_size:W1_size + b1_size].reshape(self.b1.shape)
        self.W2 = flat_params[W1_size + b1_size:W1_size + b1_size + W2_size].reshape(self.W2.shape)
        self.b2 = flat_params[W1_size + b1_size + W2_size:].reshape(self.b2.shape)