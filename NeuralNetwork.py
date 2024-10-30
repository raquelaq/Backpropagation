import numpy as np
from ActivationFunction import ActivationFunction


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation_hidden, activation_output="softmax"):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Configurar funciones de activation_hidden
        if activation_hidden == "sigmoid":
            self.activation_hidden = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation_hidden == "tanh":
            self.activation_hidden = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation_hidden == "relu":
            self.activation_hidden = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        else:
            raise ValueError("Función de activación (hidden) no soportada")
        
        if activation_output == "sigmoid":
            self.activation_output = ActivationFunction.sigmoid
            self.activation_output_derivative = ActivationFunction.sigmoid_derivative
        elif activation_output == "softmax":
            self.activation_output = ActivationFunction.softmax
            self.activation_output_derivative = ActivationFunction.softmax_derivative
        else:
            raise ValueError("Función de activación (output) no soportada")

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation_hidden(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.activation_output(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.activation_derivative(output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
