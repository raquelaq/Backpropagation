import numpy as np
from ActivationFunction import ActivationFunction


class NeuralNetwork:

    # Esta clase implementa la red neuronal descrita en la memoria.
    # Utiliza el optimizador Adam para ajustar los pesos y sesgos, proporcionando una convergencia
    # más estable y eficiente que el descenso de gradiente estocástico tradicional.
    # Los parámetros del optimizador Adam, como beta1, beta2 y epsilon, son configurables.

    # Tareas realizadas por la clase:
    #
    # 1. Inicialización: Configuración de los pesos y sesgos aleatorios y definición parámetros específicos de
    #    Adam para los momentos de primer y segundo orden.

    # 2. Propagación hacia adelante (forward): calcula las salidas de cada capa pasando las entradas a
    #    través de las conexiones y aplicando las funciones de activación correspondientes.

    # 3. Retropropagación (backward): calcula los errores de cada capa y actualiza los pesos y sesgos
    #    utilizando el optimizador Adam. En este proceso:
    #    - Se calculan los gradientes de los pesos y los sesgos de cada capa.
    #    - Se actualizan los valores de los momentos de primer y segundo orden para cada peso y sesgo.
    #    - Finalmente, los parámetros se ajustan según el gradiente y la tasa de aprendizaje.

    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, activation_hidden="sigmoid"):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Inicializar pesos, sesgos y parámetros de Adam para cada capa
        self.layers = layers
        self.weights = []
        self.biases = []
        self.m = []
        self.v = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.zeros((1, layers[i + 1])))
            self.m.append(np.zeros((layers[i], layers[i + 1])))
            self.v.append(np.zeros((layers[i], layers[i + 1])))

        # Función de activación
        if activation_hidden == "sigmoid":
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation_hidden == "tanh":
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation_hidden == "relu":
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        else:
            raise ValueError("Función de activación no soportada")

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activation(z) if i < len(self.weights) - 1 else ActivationFunction.softmax(z)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, X, y):
        output_error = self.activations[-1] - y
        deltas = [output_error]

        # Propagar hacia atrás el error
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * self.activation_derivative(self.activations[i + 1])
            deltas.append(delta)

        deltas.reverse()

        # Actualizar pesos y sesgos con Adam
        for i in range(len(self.weights)):
            d_weights = self.activations[i].T.dot(deltas[i])
            d_biases = np.sum(deltas[i], axis=0, keepdims=True)

            # Adam optimizer
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * d_weights
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (d_weights ** 2)
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)

            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * d_biases

