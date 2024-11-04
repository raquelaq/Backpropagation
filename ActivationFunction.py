import numpy as np


class ActivationFunction:

    # Esta clase define un conjunto de funciones de activación comunes utilizadas en redes neuronales.
    # Como comentamos en la memoria, las funciones de activación permiten que estas redes
    # modelen relaciones no lineales, lo cual es fundamental para su capacidad de aprendizaje.
    # Cada función está implementada como un metodo estático,
    # lo que permite llamarla sin necesidad de instanciar la clase.
    # Las funciones incluidas en esta clase son sigmoid, tanh, relu y softmax, con sus respectivas derivadas.

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - x ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x <= 0, 0, 1)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        return x * (1 - x)
