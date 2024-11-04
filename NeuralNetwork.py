import numpy as np
from ActivationFunction import ActivationFunction


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 activation_hidden="sigmoid", activation_output="softmax"):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Initialize Adam parameters
        self.m_wih = np.zeros_like(self.weights_input_hidden)
        self.v_wih = np.zeros_like(self.weights_input_hidden)
        self.m_wo = np.zeros_like(self.weights_hidden_output)
        self.v_wo = np.zeros_like(self.weights_hidden_output)
        self.m_bh = np.zeros_like(self.bias_hidden)
        self.v_bh = np.zeros_like(self.bias_hidden)
        self.m_bo = np.zeros_like(self.bias_output)
        self.v_bo = np.zeros_like(self.bias_output)

        # Activation functions
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
        output_delta = output_error * self.activation_output_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        # Calculate gradients
        d_weights_hidden_output = self.hidden_output.T.dot(output_delta)
        d_weights_input_hidden = X.T.dot(hidden_delta)
        d_bias_output = np.sum(output_delta, axis=0, keepdims=True)
        d_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)

        # Update parameters using Adam
        self.m_wih = self.beta1 * self.m_wih + (1 - self.beta1) * d_weights_input_hidden
        self.v_wih = self.beta2 * self.v_wih + (1 - self.beta2) * (d_weights_input_hidden ** 2)
        self.weights_input_hidden += self.learning_rate * self.m_wih / (np.sqrt(self.v_wih) + self.epsilon)

        self.m_wo = self.beta1 * self.m_wo + (1 - self.beta1) * d_weights_hidden_output
        self.v_wo = self.beta2 * self.v_wo + (1 - self.beta2) * (d_weights_hidden_output ** 2)
        self.weights_hidden_output += self.learning_rate * self.m_wo / (np.sqrt(self.v_wo) + self.epsilon)

        self.m_bh = self.beta1 * self.m_bh + (1 - self.beta1) * d_bias_hidden
        self.v_bh = self.beta2 * self.v_bh + (1 - self.beta2) * (d_bias_hidden ** 2)
        self.bias_hidden += self.learning_rate * self.m_bh / (np.sqrt(self.v_bh) + self.epsilon)

        self.m_bo = self.beta1 * self.m_bo + (1 - self.beta1) * d_bias_output
        self.v_bo = self.beta2 * self.v_bo + (1 - self.beta2) * (d_bias_output ** 2)
        self.bias_output += self.learning_rate * self.m_bo / (np.sqrt(self.v_bo) + self.epsilon)
