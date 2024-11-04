import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Trainer:

    # Esta clase gestiona el entrenamiento, evaluación y visualización de la pérdida de la red neuronal.
    # Simplifica el flujo de trabajo, permitiendo entrenar el modelo, evaluar su precisión y monitorear su rendimiento.

    # Principales métodos:
    # 1. train: entrena el modelo por un número específico de épocas, registrando la pérdida en cada una para
    #    observar la convergencia.
    # 2. evaluate: calcula la precisión y la matriz de confusión en el conjunto de prueba, facilitando el análisis
    #    de rendimiento del modelo.
    # 3. plot_losses: grafica la pérdida durante el entrenamiento, permitiendo una evaluación visual de la estabilidad
    #    del modelo.

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            output = self.model.forward(self.X_train)
            loss = np.mean((self.y_train - output) ** 2)
            losses.append(loss)
            self.model.backward(self.X_train, self.y_train, output)

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss}")

        return losses

    def evaluate(self):
        predictions = self.model.forward(self.X_test)
        predictions = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        conf_matrix = confusion_matrix(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        return conf_matrix, accuracy

    def plot_losses(self, losses):
        plt.figure(figsize=(5, 5))
        plt.ylim(0, 1)
        plt.plot(losses)
        plt.axhline(y=0, color='r', linewidth=1)
        plt.title("Evolución de la pérdida durante el entrenamiento")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
